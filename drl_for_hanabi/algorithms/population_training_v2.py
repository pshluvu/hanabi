"""
Population training v2 -- redesigned after the sanity checks and examiner feedback.

WHAT CHANGED vs population_training_recurrent.py, and why
----------------------------------------------------------
1. EXPLICIT HINT ACTIONS (examiner point 2).
   The old agent had ONE hint action and a random hint was drawn for it, so it could
   never choose WHAT to say and therefore could never form a hint convention.
   Now there are 20 actions:  0-4 discard | 5-9 play | 10-14 colour hint | 15-19 rank hint.
   Illegal actions are masked out of the policy (no illegal-move penalty/resampling).

2. A RICHER OBSERVATION.
   The old 136-dim vector had no view of the partner's hand and no "what did my partner
   just do" signal -- both are needed to choose and to interpret hints. This version feeds
   the environment's own full vectorised observation (obs["vectorized"]): partner's hand,
   board, discards, own hint knowledge and the LAST MOVE. The input size is read from the
   environment at run time.

3. A COMPETENT, CONVENTION-DIVERSE PARTNER POOL (examiner point 1 + sanity check).
   Dropped: simple (bombs out in every pairing), ppo / vpg / spg (cheat-trained, blind under
   standard rules). Kept: bram_simple, bram_best, hoad_piers. Added: independent SELF-PLAY
   agents trained here from different random seeds with the same 20-action space; each one
   settles on its own private convention. Self-play seeds that score below a threshold
   with themselves are dropped (a noise partner teaches nothing). Some seeds are HELD OUT:
   never trained against, used only to measure zero-shot performance.

4. PARTNER INFERENCE (the project title).
   The learner never receives the partner's identity. Its recurrent hidden state has an
   auxiliary head that tries to predict the partner type from play so far (loss only during
   training; the prediction is not an input). --oracle is the ablation that DOES give the
   partner id as an input, as an upper bound on what inference could achieve.

5. HINT ANALYSIS.  --mode eval prints, for each partner, the score and how the learner's
   hints are distributed, plus how different those distributions are across partners.

MODES
-----
  selfplay    train one self-play partner (run once per seed)
  population  train the learner against the pool
  eval        evaluate a trained learner (greedy) against train and held-out partners

Run every command with the same environment as before:
  cd drl_for_hanabi/algorithms
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 TF_ENABLE_ONEDNN_OPTS=0 \
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
  python -u population_training_v2.py --mode selfplay --seed 0
"""
import sys
import os
import argparse
import copy
import json
import math
import random
from collections import defaultdict

_ALGO_DIR = os.path.dirname(os.path.realpath(__file__))
_DRL_DIR = os.path.dirname(_ALGO_DIR)
_REPO_ROOT = os.path.dirname(_DRL_DIR)
_DECENTRALIZED_DIR = os.path.join(_DRL_DIR, "extra", "decentralized")
for _p in (_REPO_ROOT, _DECENTRALIZED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam

torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False

from hanabi_learning_environment import rl_env_adjusted
from drl_for_hanabi.utils import basics

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
NUM_ACTIONS = 20
COLORS = ["R", "Y", "G", "W", "B"]
HIDDEN_PI = 128
HIDDEN_VF = 64
LR_PI = 1e-3
LR_VF = 1e-3
GAMMA = 0.99
ENTROPY_COEF = 0.01
AUX_COEF = 0.1                   # weight of the partner-type prediction loss
BATCH_SIZE = 1000                # minimum environment steps per epoch (all seats counted)
SAVE_EVERY = 500
SHOW_EVERY = 20
GAME_CONFIG = {"colors": 5, "ranks": 5, "players": 2,
               "max_information_tokens": 8, "max_life_tokens": 3}

HAND_CODED_PARTNERS = ["bram_simple", "bram_best", "hoad_piers"]
SKIP_SELFPLAY = False            # set by --no-selfplay
SELFPLAY_TRAIN_SEEDS = [0, 1, 2, 3]
SELFPLAY_HOLDOUT_SEEDS = [4, 5]
MIN_SELFPLAY_SCORE = 3.0         # a self-play partner must at least score this with itself

PARAM_DIR = os.path.join(_DRL_DIR, "experiments", "policy_params", "v2")
SELFPLAY_DIR = os.path.join(PARAM_DIR, "selfplay")


def make_env():
    return rl_env_adjusted.make(obs_type="CARD_KNOWLEDGE", config=GAME_CONFIG)


def obs_vector(obs):
    return np.asarray(obs["vectorized"], dtype=np.float32)


def detect_obs_dim():
    env = make_env()
    obs = env.reset()["player_observations"][0]
    return int(len(obs_vector(obs)))


# ----------------------------------------------------------------------------
# Action space: 20 actions <-> environment moves
# ----------------------------------------------------------------------------
def legal_action_table(obs):
    """Returns (mask[20] bool, moves[20]) where moves[i] is what to pass to env.step()
    for action i (an int for discard/play, an HLE move dict for hints)."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    moves = [None] * NUM_ACTIONS
    legal_ints = set(obs["legal_moves_as_int"])
    for i in range(10):                       # 0-4 discard, 5-9 play
        if i in legal_ints:
            mask[i] = True
            moves[i] = i
    for m in obs["legal_moves"]:
        t = m.get("action_type")
        if t == "REVEAL_COLOR" and m.get("color") in COLORS:
            idx = 10 + COLORS.index(m["color"])
        elif t == "REVEAL_RANK" and m.get("rank") is not None and 0 <= m["rank"] < 5:
            idx = 15 + int(m["rank"])
        else:
            continue
        mask[idx] = True
        moves[idx] = m
    return mask, moves


# ----------------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------------
class RecurrentHead(nn.Module):
    """Elman-style cell built from Linear + tanh (nn.GRUCell crashes on this machine's
    PyTorch 1.7 CPU build). The hidden state is carried across the agent's own turns."""

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.hidden = hidden
        self.input_proj = nn.Linear(in_dim, hidden)
        self.combine = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x, h_prev):
        x = torch.tanh(self.input_proj(x))
        h = torch.tanh(self.combine(torch.cat([x, h_prev], dim=-1)))
        return self.out(h), h

    def init_hidden(self, batch=1):
        return torch.zeros(batch, self.hidden)


class Learner:
    """Policy + value + (optional) partner-type head.

    n_types   : number of partner types for the auxiliary head (0 = no aux head)
    oracle    : if True the one-hot partner id is APPENDED to the input (ablation only)
    """

    def __init__(self, obs_dim, n_types=0, oracle=False):
        self.obs_dim, self.n_types, self.oracle = obs_dim, n_types, oracle
        in_dim = obs_dim + (n_types if oracle else 0)
        self.policy_net = RecurrentHead(in_dim, HIDDEN_PI, NUM_ACTIONS)
        self.value_net = RecurrentHead(in_dim, HIDDEN_VF, 1)
        self.aux_head = nn.Linear(HIDDEN_PI, n_types) if (n_types and not oracle) else None
        pi_params = list(self.policy_net.parameters())
        if self.aux_head is not None:
            pi_params += list(self.aux_head.parameters())
        self.opt_pi = Adam(pi_params, lr=LR_PI)
        self.opt_vf = Adam(self.value_net.parameters(), lr=LR_VF)

    def new_hidden(self):
        return self.policy_net.init_hidden(), self.value_net.init_hidden()

    def _input(self, obs, type_id):
        x = torch.as_tensor(obs_vector(obs)).unsqueeze(0)
        if self.oracle:
            onehot = torch.zeros(1, self.n_types)
            if type_id is not None and 0 <= type_id < self.n_types:
                onehot[0, type_id] = 1.0
            x = torch.cat([x, onehot], dim=-1)
        return x

    def act(self, obs, h_pi, h_vf, explore=True, type_id=None):
        """Returns dict with the env move, action index and training quantities."""
        x = self._input(obs, type_id)
        logits, h_pi_n = self.policy_net(x, h_pi)
        value, h_vf_n = self.value_net(x, h_vf)
        mask, moves = legal_action_table(obs)
        logits = logits.squeeze(0).masked_fill(~torch.as_tensor(mask), -1e9)
        dist = Categorical(logits=logits)
        idx = int(dist.sample().item()) if explore else int(torch.argmax(logits).item())
        aux = self.aux_head(h_pi_n).squeeze(0) if self.aux_head is not None else None
        return {"idx": idx, "move": moves[idx], "log_prob": dist.log_prob(torch.as_tensor(idx)),
                "entropy": dist.entropy(), "value": value.squeeze(), "aux": aux,
                "h_pi": h_pi_n, "h_vf": h_vf_n}

    # ---- checkpoints ----
    def state(self):
        return {"policy_net": self.policy_net.state_dict(), "value_net": self.value_net.state_dict(),
                "aux_head": self.aux_head.state_dict() if self.aux_head is not None else None,
                "opt_pi": self.opt_pi.state_dict(), "opt_vf": self.opt_vf.state_dict(),
                "obs_dim": self.obs_dim, "n_types": self.n_types, "oracle": self.oracle}

    def load_state(self, ck):
        self.policy_net.load_state_dict(ck["policy_net"])
        self.value_net.load_state_dict(ck["value_net"])
        if self.aux_head is not None and ck.get("aux_head") is not None:
            self.aux_head.load_state_dict(ck["aux_head"])
        self.opt_pi.load_state_dict(ck["opt_pi"])
        self.opt_vf.load_state_dict(ck["opt_vf"])


class FrozenPolicyPartner:
    """A trained self-play agent used as a partner. Same interface as the repo's agents
    (act(obs) -> env move) plus reset() to clear the hidden state at the start of a game."""

    def __init__(self, path, explore):
        ck = torch.load(path, map_location="cpu")
        self.explore = explore
        self.net = RecurrentHead(ck["obs_dim"], HIDDEN_PI, NUM_ACTIONS)
        self.net.load_state_dict(ck["policy_net"])
        self.net.eval()
        self.reset()

    def reset(self):
        self.h = self.net.init_hidden()

    def act(self, obs):
        with torch.no_grad():
            x = torch.as_tensor(obs_vector(obs)).unsqueeze(0)
            logits, self.h = self.net(x, self.h)
            mask, moves = legal_action_table(obs)
            logits = logits.squeeze(0).masked_fill(~torch.as_tensor(mask), -1e9)
            idx = int(Categorical(logits=logits).sample().item()) if self.explore \
                else int(torch.argmax(logits).item())
        return moves[idx]


# ----------------------------------------------------------------------------
# Playing games
# ----------------------------------------------------------------------------
def final_score(observations):
    o = observations["player_observations"][0]
    fw = sum(o["fireworks"].values())
    return fw if o["life_tokens"] > 0 else 0   # repo scoring rule: bombing out scores 0


def bombed(observations):
    return observations["player_observations"][0]["life_tokens"] == 0


class ExternalPartner:
    """OBL or SAD via the repo's subprocess proxies. Partner always sits in seat 0."""
    def __init__(self, name):
        from agent_registry import build_agent
        self.name = name
        self.agent = build_agent(name)

    def close(self):
        try:
            if self.name == "obl":
                self.agent.shutdown()
            else:
                self.agent.close()
        except Exception:
            pass


def move_to_dict(obs, move):
    if isinstance(move, dict):
        return move
    return obs["legal_moves"][obs["legal_moves_as_int"].index(move)]


def play_vs_external(learner, partner, explore, type_id=None):
    from deck_sync_utils import reset_with_deck_order, step_with_deck_order
    env = make_env()
    env.reset()
    ag, pseat, seat = partner.agent, 0, 1
    if partner.name == "obl":
        ag.reset()
        observations = reset_with_deck_order(env, ag.deck_history())
    else:
        observations = reset_with_deck_order(env, ag.reset())
    h_pi, h_vf = learner.new_hidden()
    rewards, steps, actions = [], [], []
    done, guard = False, 0
    while not done and guard < 500:
        guard += 1
        cur = observations["player_observations"][0]["current_player"]
        p_move = ag.act(observations["player_observations"][pseat])
        if cur == pseat:
            assert p_move is not None, "external partner returned no move on its own turn"
            move = p_move
        else:
            obs = observations["player_observations"][seat]
            out = learner.act(obs, h_pi, h_vf, explore=explore, type_id=type_id)
            h_pi, h_vf = out["h_pi"], out["h_vf"]
            move = move_to_dict(obs, out["move"])
            steps.append({"log_prob": out["log_prob"], "entropy": out["entropy"],
                          "value": out["value"], "aux": out["aux"], "pos": len(rewards)})
            actions.append(out["idx"])
        if partner.name == "obl":
            ag.apply_external_move(move)
        else:
            ag.apply_external_move(env, move)
        observations, rew, done, _ = step_with_deck_order(env, move)
        rewards.append(rew)
    return {"steps": steps, "rewards": rewards, "score": final_score(observations),
            "bombed": bombed(observations), "actions": actions}


def play_vs_partner(learner, partner, seat, explore, type_id=None, record=False):
    """One game, learner in `seat`, partner in the other. Returns record dict."""
    if isinstance(partner, ExternalPartner):
        return play_vs_external(learner, partner, explore, type_id)
    env = make_env()
    observations = env.reset()
    if isinstance(partner, FrozenPolicyPartner):   # repo agents' reset() needs a config
        partner.reset()
    h_pi, h_vf = learner.new_hidden()
    rewards, steps, actions = [], [], []
    done, guard = False, 0
    while not done and guard < 500:
        guard += 1
        cur = observations["player_observations"][0]["current_player"]
        obs = observations["player_observations"][cur]
        if cur == seat:
            out = learner.act(obs, h_pi, h_vf, explore=explore, type_id=type_id)
            h_pi, h_vf = out["h_pi"], out["h_vf"]
            observations, rew, done, _ = env.step(out["move"])
            steps.append({"log_prob": out["log_prob"], "entropy": out["entropy"],
                          "value": out["value"], "aux": out["aux"], "pos": len(rewards)})
            actions.append(out["idx"])
        else:
            observations, rew, done, _ = env.step(partner.act(obs))
        rewards.append(rew)
    return {"steps": steps, "rewards": rewards, "score": final_score(observations),
            "bombed": bombed(observations), "actions": actions}


def play_selfplay(learner, explore=True):
    """Both seats played by the same network, each with its own hidden state."""
    env = make_env()
    observations = env.reset()
    hidden = {0: learner.new_hidden(), 1: learner.new_hidden()}
    rewards, steps = [], []
    done, guard = False, 0
    while not done and guard < 500:
        guard += 1
        cur = observations["player_observations"][0]["current_player"]
        obs = observations["player_observations"][cur]
        h_pi, h_vf = hidden[cur]
        out = learner.act(obs, h_pi, h_vf, explore=explore)
        hidden[cur] = (out["h_pi"], out["h_vf"])
        observations, rew, done, _ = env.step(out["move"])
        steps.append({"log_prob": out["log_prob"], "entropy": out["entropy"],
                      "value": out["value"], "aux": None, "pos": len(rewards)})
        rewards.append(rew)
    return {"steps": steps, "rewards": rewards, "score": final_score(observations),
            "actions": []}


# ----------------------------------------------------------------------------
# One policy-gradient update (one backward() per episode -- the PyTorch 1.7 crash fix)
# ----------------------------------------------------------------------------
def update(learner, episodes, aux_coef):
    total = sum(len(e["steps"]) for e in episodes)
    for e in episodes:
        rtg = basics.reward_to_go(e["rewards"], gamma=GAMMA)
        e["rtg"] = [rtg[s["pos"]] for s in e["steps"]]
    values_all = torch.cat([torch.stack([s["value"] for s in e["steps"]]).detach() for e in episodes])
    rtg_all = torch.as_tensor([v for e in episodes for v in e["rtg"]], dtype=torch.float32)
    adv = rtg_all - values_all
    if adv.std() > 1e-6:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    learner.opt_vf.zero_grad()
    loss_vf = 0.0
    for e in episodes:
        v = torch.stack([s["value"] for s in e["steps"]])
        r = torch.as_tensor(e["rtg"], dtype=torch.float32)
        l = ((v - r) ** 2).sum() / total
        l.backward()
        loss_vf += l.item()
    learner.opt_vf.step()

    learner.opt_pi.zero_grad()
    idx, loss_pi, ent_sum, aux_correct, aux_n = 0, 0.0, 0.0, 0, 0
    for e in episodes:
        n = len(e["steps"])
        lp = torch.stack([s["log_prob"] for s in e["steps"]])
        en = torch.stack([s["entropy"] for s in e["steps"]])
        l = (-(lp * adv[idx: idx + n]).sum() - ENTROPY_COEF * en.sum()) / total
        if learner.aux_head is not None and e.get("type_id") is not None:
            aux_logits = torch.stack([s["aux"] for s in e["steps"]])
            target = torch.full((n,), e["type_id"], dtype=torch.long)
            l = l + aux_coef * F.cross_entropy(aux_logits, target, reduction="sum") / total
            aux_correct += int((aux_logits.argmax(-1) == target).sum().item())
            aux_n += n
        l.backward()
        loss_pi += l.item()
        ent_sum += en.sum().item()
        idx += n
    learner.opt_pi.step()
    return {"loss_pi": loss_pi, "loss_vf": loss_vf, "entropy": ent_sum / total,
            "aux_acc": (aux_correct / aux_n) if aux_n else float("nan")}


# ----------------------------------------------------------------------------
# Partner pool
# ----------------------------------------------------------------------------
def selfplay_path(seed):
    return os.path.join(SELFPLAY_DIR, f"sp_seed{seed}.pth")


def selfplay_score(path, games=20):
    """Mean score of a self-play partner playing a copy of itself (greedy)."""
    a, b = FrozenPolicyPartner(path, explore=False), FrozenPolicyPartner(path, explore=False)
    scores = []
    for _ in range(games):
        env = make_env()
        observations = env.reset()
        a.reset()
        b.reset()
        agents = {0: a, 1: b}
        done, guard = False, 0
        while not done and guard < 500:
            guard += 1
            cur = observations["player_observations"][0]["current_player"]
            observations, _, done, _ = env.step(agents[cur].act(observations["player_observations"][cur]))
        scores.append(final_score(observations))
    return float(np.mean(scores))


def build_pool(min_score=MIN_SELFPLAY_SCORE, explore_selfplay=True, hand_coded=None):
    """Returns (train_pool, holdout_pool): dicts name -> agent."""
    from agent_registry import build_agent
    train, holdout = {}, {}
    for name in (hand_coded if hand_coded is not None else HAND_CODED_PARTNERS):
        try:
            train[name] = build_agent(name)
            print(f"pool: hand-coded partner '{name}' loaded")
        except Exception as e:  # noqa: BLE001
            print(f"pool: SKIPPED '{name}' ({e})")
    for target, seeds in (() if SKIP_SELFPLAY else
                          ((train, SELFPLAY_TRAIN_SEEDS), (holdout, SELFPLAY_HOLDOUT_SEEDS))):
        for s in seeds:
            p = selfplay_path(s)
            if not os.path.exists(p):
                print(f"pool: self-play seed {s} not trained yet -- skipped")
                continue
            sc = selfplay_score(p)
            if sc < min_score:
                print(f"pool: self-play seed {s} DROPPED (self-play score {sc:.2f} < {min_score})")
                continue
            target[f"sp{s}"] = FrozenPolicyPartner(p, explore=explore_selfplay)
            print(f"pool: self-play seed {s} loaded (self-play score {sc:.2f})")
    if not train:
        raise RuntimeError("empty training pool")
    return train, holdout


# ----------------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------------
def latest_checkpoint(run_dir):
    best, path = 0, None
    if os.path.isdir(run_dir):
        for f in os.listdir(run_dir):
            if f.startswith("policy_epoch") and f.endswith(".pth"):
                try:
                    n = int(f[len("policy_epoch"):-4])
                except ValueError:
                    continue
                if n > best:
                    best, path = n, os.path.join(run_dir, f)
    return path, best


def save(learner, path, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    st = learner.state()
    if extra:
        st.update(extra)
    torch.save(st, path)
    print(f"saved {path}")


# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------
def run_selfplay(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    obs_dim = detect_obs_dim()
    learner = Learner(obs_dim)
    run_dir = os.path.join(SELFPLAY_DIR, f"run_seed{args.seed}")
    path, start = latest_checkpoint(run_dir)
    if path:
        learner.load_state(torch.load(path, map_location="cpu"))
        print(f"resumed seed {args.seed} from epoch {start}")
    for epoch in range(start + 1, args.epochs + 1):
        episodes, steps = [], 0
        while steps < BATCH_SIZE:
            ep = play_selfplay(learner)
            episodes.append(ep)
            steps += len(ep["steps"])
        stats = update(learner, episodes, 0.0)
        if epoch % SHOW_EVERY == 0:
            print(f"[sp seed {args.seed}] epoch {epoch:6d} | games {len(episodes):3d} | "
                  f"score {np.mean([e['score'] for e in episodes]):5.2f} | "
                  f"entropy {stats['entropy']:.3f} | loss_pi {stats['loss_pi']:.4f}", flush=True)
        if epoch % SAVE_EVERY == 0:
            save(learner, os.path.join(run_dir, f"policy_epoch{epoch}.pth"))
            save(learner, selfplay_path(args.seed))      # always-current partner file
    save(learner, selfplay_path(args.seed))
    print(f"self-play seed {args.seed} done; self-play score "
          f"{selfplay_score(selfplay_path(args.seed)):.2f}")


def run_population(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    obs_dim = detect_obs_dim()
    train_pool, holdout_pool = build_pool(hand_coded=args.partners)
    for n in (args.train_external or []):
        train_pool[n] = ExternalPartner(n)
        print(f"pool: external training partner '{n}' loaded (seat 0)")
    names = list(train_pool.keys())
    print(f"training partners: {names} | held out: {list(holdout_pool.keys())}")
    learner = Learner(obs_dim, n_types=len(names), oracle=args.oracle)
    run_dir = os.path.join(PARAM_DIR, args.run_name)
    path, start = latest_checkpoint(run_dir)
    if not path and args.init_from:
        ck0 = torch.load(args.init_from, map_location="cpu")
        learner.policy_net.load_state_dict(ck0["policy_net"])
        learner.value_net.load_state_dict(ck0["value_net"])
        print(f"initialised policy/value nets from {args.init_from} (aux head + optimisers fresh)")
    if path:
        ck = torch.load(path, map_location="cpu")
        if ck.get("partner_names") != names:
            raise RuntimeError(f"pool changed since checkpoint: {ck.get('partner_names')} vs {names}")
        learner.load_state(ck)
        print(f"resumed from epoch {start}")
    meta = {"partner_names": names, "holdout_names": list(holdout_pool.keys())}
    aux_coef = 0.0 if args.no_aux else AUX_COEF
    for epoch in range(start + 1, args.epochs + 1):
        episodes, steps = [], 0
        per_partner = defaultdict(list)
        while steps < BATCH_SIZE:
            name = random.choice(names)
            type_id = names.index(name)
            try:
                ep = play_vs_partner(learner, train_pool[name], seat=random.randint(0, 1),
                                     explore=True, type_id=type_id)
            except Exception as e:
                if not isinstance(train_pool[name], ExternalPartner):
                    raise
                print(f"external partner '{name}' failed ({e}); restarting it", flush=True)
                train_pool[name].close()
                train_pool[name] = ExternalPartner(name)
                continue
            ep["type_id"] = type_id
            episodes.append(ep)
            per_partner[name].append(ep["score"])
            steps += len(ep["steps"])
        stats = update(learner, episodes, aux_coef)
        if epoch % SHOW_EVERY == 0:
            by = " ".join(f"{n}:{np.mean(v):.1f}" for n, v in sorted(per_partner.items()))
            print(f"epoch {epoch:6d} | games {len(episodes):3d} | score "
                  f"{np.mean([e['score'] for e in episodes]):5.2f} | entropy {stats['entropy']:.3f} | "
                  f"aux_acc {stats['aux_acc']:.2f} | {by}", flush=True)
        if epoch % SAVE_EVERY == 0:
            save(learner, os.path.join(run_dir, f"policy_epoch{epoch}.pth"), meta)
    save(learner, os.path.join(run_dir, "policy_final.pth"), meta)
    for p in train_pool.values():
        if isinstance(p, ExternalPartner):
            p.close()


def jensen_shannon(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run_eval(args):
    obs_dim = detect_obs_dim()
    run_dir = os.path.join(PARAM_DIR, args.run_name)
    ck = torch.load(os.path.join(run_dir, "policy_final.pth") if not args.checkpoint else args.checkpoint,
                    map_location="cpu")
    names = ck["partner_names"]
    learner = Learner(obs_dim, n_types=len(names), oracle=ck.get("oracle", False))
    learner.load_state(ck)
    train_pool, holdout_pool = build_pool(explore_selfplay=False, hand_coded=args.partners)
    rows, hint_dists = [], {}
    ext_pool = {}
    for n in (args.external or []):
        try:
            ext_pool[n] = ExternalPartner(n)
            print(f"pool: external partner '{n}' loaded (seat 0)")
        except Exception as e:
            print(f"pool: SKIPPED external '{n}' ({e})")
    for group, pool in (("train", train_pool), ("HELD-OUT", holdout_pool), ("EXTERNAL", ext_pool)):
        for name, agent in pool.items():
            type_id = names.index(name) if name in names else None
            group_label = "UNSEEN" if (group == "train" and type_id is None) else group
            scores, hints, bombs = [], np.zeros(10), 0
            with torch.no_grad():
                for g in range(args.eval_episodes):
                    ep = play_vs_partner(learner, agent, seat=g % 2, explore=False,
                                         type_id=type_id)
                    scores.append(ep["score"])
                    bombs += int(ep["bombed"])
                    for a in ep["actions"]:
                        if a >= 10:
                            hints[a - 10] += 1
            rows.append((group_label, name, np.mean(scores), np.std(scores)))
            if hints.sum() > 0:
                hint_dists[name] = hints / hints.sum()
            print(f"{group_label:9s} {name:12s} score {np.mean(scores):6.2f} +- {np.std(scores):5.2f}"
                  f"   bombed {bombs:2d}/{args.eval_episodes}"
                  f"   hints/game {hints.sum() / args.eval_episodes:5.2f}"
                  f"   colour {np.round(hints[:5] / max(1, hints.sum()), 2)}"
                  f"   rank {np.round(hints[5:] / max(1, hints.sum()), 2)}", flush=True)
    keys = list(hint_dists)
    if len(keys) > 1:
        js = [jensen_shannon(hint_dists[a], hint_dists[b])
              for i, a in enumerate(keys) for b in keys[i + 1:]]
        print(f"\nmean pairwise Jensen-Shannon divergence of hint distributions across partners: "
              f"{np.mean(js):.4f}  (0 = same hints for everyone; ln2 = 0.693 = completely different)")
    train_scores = [r[2] for r in rows if r[0] == "train"]
    held = [r[2] for r in rows if r[0] in ("HELD-OUT", "UNSEEN")]
    ext = [r[2] for r in rows if r[0] == "EXTERNAL"]
    for p in ext_pool.values():
        p.close()
    if train_scores:
        print(f"mean over training partners: {np.mean(train_scores):.2f}")
    if held:
        print(f"mean over HELD-OUT partners (zero-shot): {np.mean(held):.2f}")
    if ext:
        print(f"mean over EXTERNAL partners (zero-shot): {np.mean(ext):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selfplay", "population", "eval"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--run-name", default="population")
    ap.add_argument("--oracle", action="store_true", help="ablation: give the learner the partner id")
    ap.add_argument("--no-aux", action="store_true", help="ablation: disable the partner-type loss")
    ap.add_argument("--partners", nargs="+", default=None)
    ap.add_argument("--external", nargs="+", default=None, choices=["obl", "sad"],
                    help="eval only: also play against OBL and/or SAD (zero-shot)")
    ap.add_argument("--train-external", nargs="+", default=None, choices=["obl", "sad"],
                    help="population mode: also TRAIN against OBL and/or SAD")
    ap.add_argument("--no-selfplay", action="store_true",
                    help="leave the self-play seeds out of the pool (fixed, reproducible pool)")
    ap.add_argument("--init-from", default=None,
                    help="population mode: start the policy/value nets from this checkpoint")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--eval-episodes", type=int, default=50)
    args = ap.parse_args()
    globals()["SKIP_SELFPLAY"] = args.no_selfplay
    if args.epochs is None:
        args.epochs = {"selfplay": 30000, "population": 100000, "eval": 0}[args.mode]
    {"selfplay": run_selfplay, "population": run_population, "eval": run_eval}[args.mode](args)


if __name__ == "__main__":
    main()
