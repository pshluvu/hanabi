"""
Population training v3 = v2 + (a) tunable BAD-style hyper-parameters, (b) an optional
privileged critic, (c) a CLEAR-style replay buffer that keeps experience from earlier
partners when new partners are added.  Everything is OFF/identical to v2 by default.

NEW FLAGS (population mode)
  --gamma 0.999 --lr 3e-4 --entropy 0.02     hyper-parameters taken from the BAD paper
  --hidden-pi 256 --hidden-vf 256            bigger recurrent cells
  --privileged-critic                        value net also sees the agent's OWN hand (training only)
  --replay-cap 3000                          keep up to 3000 past games (reservoir sampling)
  --replay-from path/replay_buffer.pt        start from the stored games of an earlier stage
  --replay-mix 0.5 --bc-policy 1.0 --bc-value 0.1   how strongly old behaviour is held in place

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

PRIV_DIM = 5 * 25                # own hand, one-hot (5 cards x 25 colour/rank combinations)
USE_PRIV = False                 # set by --privileged-critic
REPLAY_MIX = 0.5                 # replayed steps per new step
BC_POLICY = 1.0                  # weight of KL(old policy || current policy) on replayed games
BC_VALUE = 0.1                   # weight of (V - V_old)^2 on replayed games

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


def privileged_vec(env, seat):
    """The agent's OWN hand (hidden from the agent, visible to the simulator) as 5x25 one-hot.
    Used only as an extra input to the value network during training (centralised critic)."""
    v = np.zeros(PRIV_DIM, dtype=np.float32)
    try:
        hand = env.state.player_hands()[seat]
        for i, card in enumerate(hand[:5]):
            c, r = card.color(), card.rank()
            if c is not None and r is not None and 0 <= c < 5 and 0 <= r < 5:
                v[i * 25 + c * 5 + r] = 1.0
    except Exception:                           # noqa: BLE001
        pass
    return v


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

    def __init__(self, obs_dim, n_types=0, oracle=False, priv=False):
        self.obs_dim, self.n_types, self.oracle, self.priv = obs_dim, n_types, oracle, priv
        self.trace = False                      # when True, act() also returns what replay needs
        in_dim = obs_dim + (n_types if oracle else 0)
        self.policy_net = RecurrentHead(in_dim, HIDDEN_PI, NUM_ACTIONS)
        self.value_net = RecurrentHead(in_dim + (PRIV_DIM if priv else 0), HIDDEN_VF, 1)
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

    def act(self, obs, h_pi, h_vf, explore=True, type_id=None, extra=None):
        """Returns dict with the env move, action index and training quantities."""
        x = self._input(obs, type_id)
        logits, h_pi_n = self.policy_net(x, h_pi)
        xv = x
        if self.priv:
            e = torch.as_tensor(extra, dtype=torch.float32).unsqueeze(0) if extra is not None \
                else torch.zeros(1, PRIV_DIM)
            xv = torch.cat([x, e], dim=-1)
        value, h_vf_n = self.value_net(xv, h_vf)
        mask, moves = legal_action_table(obs)
        logits = logits.squeeze(0).masked_fill(~torch.as_tensor(mask), -1e9)
        dist = Categorical(logits=logits)
        idx = int(dist.sample().item()) if explore else int(torch.argmax(logits).item())
        aux = self.aux_head(h_pi_n).squeeze(0) if self.aux_head is not None else None
        trace = None
        if self.trace:
            trace = {"obs": x.squeeze(0).detach().numpy().copy(),
                     "extra": (np.asarray(extra, dtype=np.float32) if (self.priv and extra is not None) else None),
                     "mask": mask.copy(),
                     "probs": torch.softmax(logits, dim=-1).detach().numpy().copy(),
                     "value": float(value.item())}
        return {"idx": idx, "move": moves[idx], "log_prob": dist.log_prob(torch.as_tensor(idx)),
                "entropy": dist.entropy(), "value": value.squeeze(), "aux": aux,
                "h_pi": h_pi_n, "h_vf": h_vf_n, "trace": trace}

    # ---- checkpoints ----
    def state(self):
        return {"policy_net": self.policy_net.state_dict(), "value_net": self.value_net.state_dict(),
                "aux_head": self.aux_head.state_dict() if self.aux_head is not None else None,
                "opt_pi": self.opt_pi.state_dict(), "opt_vf": self.opt_vf.state_dict(),
                "obs_dim": self.obs_dim, "n_types": self.n_types, "oracle": self.oracle,
                "priv": self.priv, "hidden_pi": HIDDEN_PI, "hidden_vf": HIDDEN_VF}

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
    """OBL or SAD, reached through the repo's subprocess proxies. They need the environment's
    deck order synchronised with their own engine, so they use their own game loop
    (play_vs_external). The proxies are configured with player_idx=0, so the partner ALWAYS
    sits in seat 0 and the learner in seat 1."""
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
        except Exception:                       # noqa: BLE001
            pass


def move_to_dict(obs, move):
    """Learner moves are ints (play/discard) or dicts (hints); the proxies want dicts."""
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
        p_move = ag.act(observations["player_observations"][pseat])   # every turn, like play_together
        if cur == pseat:
            assert p_move is not None, "external partner returned no move on its own turn"
            move = p_move
        else:
            obs = observations["player_observations"][seat]
            out = learner.act(obs, h_pi, h_vf, explore=explore, type_id=type_id,
                              extra=(privileged_vec(env, seat) if USE_PRIV else None))
            h_pi, h_vf = out["h_pi"], out["h_vf"]
            move = move_to_dict(obs, out["move"])
            steps.append({"log_prob": out["log_prob"], "entropy": out["entropy"],
                          "value": out["value"], "aux": out["aux"], "pos": len(rewards),
                          "trace": out["trace"]})
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
    if isinstance(partner, FrozenPolicyPartner):   # repo agents have their own reset(config); leave them alone
        partner.reset()
    h_pi, h_vf = learner.new_hidden()
    rewards, steps, actions = [], [], []
    done, guard = False, 0
    while not done and guard < 500:
        guard += 1
        cur = observations["player_observations"][0]["current_player"]
        obs = observations["player_observations"][cur]
        if cur == seat:
            out = learner.act(obs, h_pi, h_vf, explore=explore, type_id=type_id,
                              extra=(privileged_vec(env, seat) if USE_PRIV else None))
            h_pi, h_vf = out["h_pi"], out["h_vf"]
            observations, rew, done, _ = env.step(out["move"])
            steps.append({"log_prob": out["log_prob"], "entropy": out["entropy"],
                          "value": out["value"], "aux": out["aux"], "pos": len(rewards),
                          "trace": out["trace"]})
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
# Replay buffer (CLEAR-style): keeps whole games so that old partners are not forgotten
# ----------------------------------------------------------------------------
class ReplayBuffer:
    """Reservoir-sampled store of the learner's own turns. Each stored game keeps what is
    needed to re-run the recurrent policy exactly: observations, legal-move masks, the old
    policy's action probabilities and the old value estimates."""

    def __init__(self, cap):
        self.cap, self.eps, self.seen = cap, [], 0

    def __len__(self):
        return len(self.eps)

    @staticmethod
    def pack(steps):
        tr = [s["trace"] for s in steps if s.get("trace") is not None]
        if len(tr) < 2:
            return None
        obs = np.stack([t["obs"] for t in tr])
        if np.array_equal(obs, obs.astype(bool)):
            obs = obs.astype(np.uint8)
        else:
            obs = obs.astype(np.float16)
        extra = None
        if all(t["extra"] is not None for t in tr):
            extra = np.stack([t["extra"] for t in tr]).astype(np.uint8)
        return {"obs": obs, "extra": extra,
                "mask": np.stack([t["mask"] for t in tr]).astype(bool),
                "probs": np.stack([t["probs"] for t in tr]).astype(np.float16),
                "value": np.asarray([t["value"] for t in tr], dtype=np.float32)}

    def add(self, steps):
        ep = self.pack(steps)
        if ep is None:
            return
        self.seen += 1
        if len(self.eps) < self.cap:
            self.eps.append(ep)
        else:
            j = random.randrange(self.seen)
            if j < self.cap:
                self.eps[j] = ep

    def sample_steps(self, n_steps):
        """Random whole games until at least n_steps learner steps are collected."""
        out, got = [], 0
        if not self.eps:
            return out
        while got < n_steps:
            e = random.choice(self.eps)
            out.append(e)
            got += len(e["value"])
        return out

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"cap": self.cap, "eps": self.eps, "seen": self.seen}, path)

    def load(self, path):
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)   # newer PyTorch
        except TypeError:
            d = torch.load(path, map_location="cpu")                        # PyTorch 1.7
        self.eps, self.seen = d["eps"][: self.cap], d["seen"]


def replay_backward(learner, eps, bc_policy, bc_value):
    """Adds gradients that keep the CURRENT networks close to the networks that played the
    stored games (policy: KL(old||new); value: squared difference). One backward() per game."""
    total = sum(len(e["value"]) for e in eps)
    kl_sum = 0.0
    for e in eps:
        x = torch.as_tensor(e["obs"].astype(np.float32))
        ex = torch.as_tensor(e["extra"].astype(np.float32)) if e["extra"] is not None else None
        mask = torch.as_tensor(e["mask"])
        p_old = torch.as_tensor(e["probs"].astype(np.float32))
        v_old = torch.as_tensor(e["value"])
        h_pi, h_vf = learner.new_hidden()
        kls, vds = [], []
        for t in range(x.shape[0]):
            xt = x[t: t + 1]
            logits, h_pi = learner.policy_net(xt, h_pi)
            logits = logits.squeeze(0).masked_fill(~mask[t], -1e9)
            logp = F.log_softmax(logits, dim=-1)
            po = p_old[t]
            kls.append((po * (torch.log(po.clamp_min(1e-8)) - logp)).sum())
            xv = xt
            if learner.priv:
                xv = torch.cat([xt, (ex[t: t + 1] if ex is not None else torch.zeros(1, PRIV_DIM))], dim=-1)
            v, h_vf = learner.value_net(xv, h_vf)
            vds.append((v.squeeze() - v_old[t]) ** 2)
        kl, vd = torch.stack(kls).sum(), torch.stack(vds).sum()
        (bc_policy * kl + bc_value * vd).div(total).backward()
        kl_sum += kl.item()
    return kl_sum / total


# ----------------------------------------------------------------------------
# One policy-gradient update (one backward() per episode -- the PyTorch 1.7 crash fix)
# ----------------------------------------------------------------------------
def update(learner, episodes, aux_coef, replay_eps=None):
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
    bc_kl = float("nan")
    if replay_eps:
        bc_kl = replay_backward(learner, replay_eps, BC_POLICY, BC_VALUE)
    learner.opt_vf.step()
    learner.opt_pi.step()
    return {"loss_pi": loss_pi, "loss_vf": loss_vf, "entropy": ent_sum / total,
            "aux_acc": (aux_correct / aux_n) if aux_n else float("nan"), "bc_kl": bc_kl}


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
    if args.replay_cap and args.oracle:
        raise RuntimeError("replay is not supported together with --oracle")
    learner = Learner(obs_dim, n_types=len(names), oracle=args.oracle, priv=USE_PRIV)
    run_dir = os.path.join(PARAM_DIR, args.run_name)
    path, start = latest_checkpoint(run_dir)
    if not path and args.init_from:
        ck0 = torch.load(args.init_from, map_location="cpu")
        for key, net in (("policy_net", learner.policy_net), ("value_net", learner.value_net)):
            try:
                net.load_state_dict(ck0[key])
                print(f"initialised {key} from {args.init_from}")
            except RuntimeError as e:
                print(f"{key} NOT loaded (shape differs, starts fresh): {str(e)[:90]}")
        print("aux head + optimisers start fresh")
    buf = None
    if args.replay_cap:
        buf = ReplayBuffer(args.replay_cap)
        learner.trace = True
        own = os.path.join(run_dir, "replay_buffer.pt")
        for p in ([own] if (path and os.path.exists(own)) else []) + \
                 ([args.replay_from] if args.replay_from and not (path and os.path.exists(own)) else []):
            buf.load(p)
            print(f"replay buffer: loaded {len(buf)} stored games from {p}")
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
        replay_eps = buf.sample_steps(int(REPLAY_MIX * steps)) if buf is not None else None
        stats = update(learner, episodes, aux_coef, replay_eps)
        if buf is not None:
            for ep in episodes:
                buf.add(ep["steps"])
        if epoch % SHOW_EVERY == 0:
            by = " ".join(f"{n}:{np.mean(v):.1f}" for n, v in sorted(per_partner.items()))
            print(f"epoch {epoch:6d} | games {len(episodes):3d} | score "
                  f"{np.mean([e['score'] for e in episodes]):5.2f} | entropy {stats['entropy']:.3f} | "
                  f"aux_acc {stats['aux_acc']:.2f} | "
                  + (f"bc_kl {stats['bc_kl']:.3f} buf {len(buf)} | " if buf is not None else "")
                  + f"{by}", flush=True)
        if epoch % SAVE_EVERY == 0:
            save(learner, os.path.join(run_dir, f"policy_epoch{epoch}.pth"), meta)
            if buf is not None:
                buf.save(os.path.join(run_dir, "replay_buffer.pt"))
    save(learner, os.path.join(run_dir, "policy_final.pth"), meta)
    if buf is not None:
        buf.save(os.path.join(run_dir, "replay_buffer.pt"))
        print(f"replay buffer saved: {len(buf)} games")
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
    globals()["HIDDEN_PI"] = ck.get("hidden_pi", HIDDEN_PI)
    globals()["HIDDEN_VF"] = ck.get("hidden_vf", HIDDEN_VF)
    learner = Learner(obs_dim, n_types=len(names), oracle=ck.get("oracle", False),
                      priv=ck.get("priv", False))
    learner.load_state(ck)
    train_pool, holdout_pool = build_pool(explore_selfplay=False, hand_coded=args.partners)
    rows, hint_dists = [], {}
    ext_pool = {}
    self_pool = {}
    if args.selfplay_eval:
        p = args.checkpoint or os.path.join(run_dir, "policy_final.pth")
        self_pool["self(copy)"] = FrozenPolicyPartner(p, explore=False)
        print("pool: self-play copy loaded (evaluation only)")
    for n in (args.external or []):
        try:
            ext_pool[n] = ExternalPartner(n)
            print(f"pool: external partner '{n}' loaded (seat 0)")
        except Exception as e:                  # noqa: BLE001
            print(f"pool: SKIPPED external '{n}' ({e})")
    for group, pool in (("train", train_pool), ("HELD-OUT", holdout_pool), ("EXTERNAL", ext_pool),
                        ("SELF-PLAY", self_pool)):
        for name, agent in pool.items():
            type_id = names.index(name) if name in names else None
            if group == "train" and type_id is None:
                group_label = "UNSEEN"      # not a training partner of this checkpoint -> zero-shot
            else:
                group_label = group
            if group == "EXTERNAL" and type_id is not None:
                group_label = "train(ext)"
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
    ap.add_argument("--partners", nargs="+", default=None,
                    help="hand-coded partners to use (default: bram_simple bram_best hoad_piers)")
    ap.add_argument("--external", nargs="+", default=None, choices=["obl", "sad"],
                    help="eval only: also play against OBL and/or SAD (zero-shot)")
    ap.add_argument("--train-external", nargs="+", default=None, choices=["obl", "sad"],
                    help="population mode: also TRAIN against OBL and/or SAD")
    ap.add_argument("--no-selfplay", action="store_true",
                    help="leave the self-play seeds out of the pool (fixed, reproducible pool)")
    ap.add_argument("--init-from", default=None,
                    help="population mode: start the policy/value nets from this checkpoint")
    ap.add_argument("--gamma", type=float, default=GAMMA, help="discount (BAD paper: 0.999)")
    ap.add_argument("--lr", type=float, default=LR_PI, help="learning rate (BAD paper: 1e-4..4e-4)")
    ap.add_argument("--entropy", type=float, default=ENTROPY_COEF, help="entropy weight (BAD: 0.01..0.07)")
    ap.add_argument("--hidden-pi", type=int, default=HIDDEN_PI)
    ap.add_argument("--hidden-vf", type=int, default=HIDDEN_VF)
    ap.add_argument("--privileged-critic", action="store_true",
                    help="value net also sees the agent's own hand (training only)")
    ap.add_argument("--replay-cap", type=int, default=0, help="games kept in the replay buffer (0 = off)")
    ap.add_argument("--replay-from", default=None, help="replay_buffer.pt of an earlier stage")
    ap.add_argument("--replay-mix", type=float, default=REPLAY_MIX)
    ap.add_argument("--bc-policy", type=float, default=BC_POLICY)
    ap.add_argument("--bc-value", type=float, default=BC_VALUE)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--selfplay-eval", action="store_true")
    ap.add_argument("--eval-episodes", type=int, default=50)
    args = ap.parse_args()
    globals()["SKIP_SELFPLAY"] = args.no_selfplay
    for k, v in (("GAMMA", args.gamma), ("LR_PI", args.lr), ("LR_VF", args.lr),
                 ("ENTROPY_COEF", args.entropy), ("HIDDEN_PI", args.hidden_pi),
                 ("HIDDEN_VF", args.hidden_vf), ("USE_PRIV", args.privileged_critic),
                 ("REPLAY_MIX", args.replay_mix), ("BC_POLICY", args.bc_policy),
                 ("BC_VALUE", args.bc_value)):
        globals()[k] = v
    if args.epochs is None:
        args.epochs = {"selfplay": 30000, "population": 100000, "eval": 0}[args.mode]
    {"selfplay": run_selfplay, "population": run_population, "eval": run_eval}[args.mode](args)


if __name__ == "__main__":
    main()
