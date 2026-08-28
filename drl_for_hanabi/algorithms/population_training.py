"""
Population-based training for a Hanabi agent that can play with ANY of the
existing agents in this repo, not just a copy of itself.

WHY THIS IS DIFFERENT FROM ppo.py / vpg.py / spg.py:
Those scripts train via self-play: one shared policy plays BOTH seats of a
2-player game against itself. That produces strong play between two copies
of the *same* policy, but the policy develops private conventions that don't
transfer well to a different partner -- which is exactly what you saw when
mixing ppo/vpg/spg with each other or with rule-based agents.

This script instead trains ONE learner (seat 0) against a PARTNER sampled
randomly, every episode, from a pool built out of your existing agents
(seat 1). Because the learner never knows in advance who it's paired with,
it's pushed toward conventions that work reasonably well with many
different partners instead of overfitting to itself. This is the standard
"population play" / ad-hoc teamwork recipe.

Algorithm: vanilla policy gradient (REINFORCE) with a learned value baseline,
using reward-to-go computed over the WHOLE episode (both seats' rewards,
since Hanabi's score is a shared team reward) but only applied to gradient
updates at the learner's OWN decision points.

Usage:
    cd drl_for_hanabi/algorithms
    CUDA_VISIBLE_DEVICES="" nohup python population_training.py \
        > population_training_log.txt 2>&1 &
    tail -f population_training_log.txt

This is a new, from-scratch script (not a copy of an existing one) so run
it and iterate -- treat the numbers below as a reasonable starting point,
not a tuned final config.
"""
import sys
import os

# Make the repo importable regardless of where this script is run from.
_ALGO_DIR = os.path.dirname(os.path.realpath(__file__))
_DRL_DIR = os.path.dirname(_ALGO_DIR)  # drl_for_hanabi/
_REPO_ROOT = os.path.dirname(_DRL_DIR)  # repo root, contains hanabi_learning_environment
_DECENTRALIZED_DIR = os.path.join(_DRL_DIR, "extra", "decentralized")
for _p in (_REPO_ROOT, _DECENTRALIZED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from hanabi_learning_environment import rl_env_adjusted
from drl_for_hanabi.utils import basics, state_representation_dec, action_representation_dec
from agent_registry import build_agent  # noqa: E402  (lives in extra/decentralized/)


# ----------------------------------------------------------------------------
# Config -- tune these. Values below are a reasonable starting point, not a
# tuned final config; watch the printed score and adjust as needed.
# ----------------------------------------------------------------------------
EPOCHS = 500                 # number of training epochs; bump way up for a real run
BATCH_SIZE = 1000            # minimum number of environment steps per epoch
OBS_VEC_TYPE = 136           # matches VPG's encoding: fireworks, own hint-knowledge, discards
NUM_ACTIONS = 11             # 0-4 discard, 5-9 play, 10 hint (same scheme as existing agents)
HIDDEN_LAYERS_PI = [128, 128, 64]
HIDDEN_LAYERS_VF = [128, 64, 32]
LR_PI = 1e-3
LR_VF = 1e-3
GAMMA = 0.99
ILLEGAL_MOVE_PENALTY = -1.0
ILLEGAL_MOVE_MAX = 50        # safety cap: abandon episode if this many illegal picks happen
SAVE_PARAMS_EVERY = 50       # epochs between checkpoint saves (0 = only save at the very end)
SHOW_EPOCH_EVERY = 5

# Partners to sample from each episode. Keep this pool restricted to agents
# that are actually reasonable teammates -- 'random' and 'bram_bias' are
# deliberately bad/chaotic and would teach the learner to expect nonsense.
PARTNER_NAMES = ["simple", "bram_simple", "bram_best", "ppo", "vpg", "spg"]

SAVE_DIR = os.path.join(_DRL_DIR, "experiments", "policy_params",
                         f"population_{datetime.now().strftime('%Y-%m-%d')}")


def build_partner_pool():
    """Builds one instance of each partner agent, once, up front.

    Rebuilding an RL partner from its checkpoint every episode would mean
    hitting disk every episode -- building once and reusing is much faster.
    Rule-based partners are stateless enough to reuse safely across episodes
    too (their reset(), if any, is a no-op or purely internal).
    """
    pool = {}
    for name in PARTNER_NAMES:
        try:
            pool[name] = build_agent(name)
            print(f"Partner pool: loaded '{name}'")
        except Exception as e:  # noqa: BLE001
            print(f"Partner pool: SKIPPED '{name}' (could not build: {e})")
    if not pool:
        raise RuntimeError("No partners could be built -- check PARTNER_NAMES / agent_registry.")
    return pool


class PopulationAgent:
    """The trainable learner. Architecture mirrors VPG/PPO's dec agents."""

    def __init__(self):
        self.device = basics.get_default_device()
        self.obs_to_vec_func = state_representation_dec.obs_to_vec136_binary
        self.policy_network = basics.neural_network(
            OBS_VEC_TYPE, HIDDEN_LAYERS_PI, NUM_ACTIONS, nn.Tanh)
        self.value_network = basics.neural_network(
            OBS_VEC_TYPE, HIDDEN_LAYERS_VF, 1, nn.Tanh)
        self.policy_network.to(self.device)
        self.value_network.to(self.device)
        self.optimizer_pi = Adam(self.policy_network.parameters(), lr=LR_PI)
        self.optimizer_vf = Adam(self.value_network.parameters(), lr=LR_VF)

    def observation_to_vector(self, obs):
        return self.obs_to_vec_func(obs)

    def act_int_and_legal(self, obs, explore=True):
        """Pick a discrete action integer (0-10) and report if it's legal.

        This intentionally mirrors choose_action_11position's contract, but
        is written out directly here (rather than calling that function)
        because that function returns an HLE action dict, and we also need
        the raw log-prob for the policy-gradient update -- easier to keep
        both concerns in one place for a from-scratch training script.
        """
        obs_vec = self.observation_to_vector(obs)
        obs_vec_t = torch.as_tensor(obs_vec, dtype=torch.float32).to(self.device)
        logits = self.policy_network(obs_vec_t)
        dist = Categorical(logits=logits)
        act_int = dist.sample().item() if explore else torch.argmax(dist.probs).item()
        legals_int = obs['legal_moves_as_int']
        legals_HLE = obs['legal_moves']
        hint_possible = (obs['information_tokens'] > 0)
        legal = False
        action = None
        if act_int == 10:
            if hint_possible:
                action = action_representation_dec.get_random_hint(legals_HLE)
                legal = True
        elif act_int in legals_int:
            action = act_int
            legal = True
        return obs_vec, act_int, action, legal, dist.log_prob(torch.as_tensor(act_int).to(self.device))

    def value_estimate(self, obs_vec):
        obs_vec_t = torch.as_tensor(obs_vec, dtype=torch.float32).to(self.device)
        return self.value_network(obs_vec_t)

    def save_policy(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "policy_network": self.policy_network.state_dict(),
            "optimizer_pi": self.optimizer_pi.state_dict(),
            "value_network": self.value_network.state_dict(),
            "optimizer_vf": self.optimizer_vf.state_dict(),
        }, save_path)
        print(f"Saved checkpoint to {save_path}")


def run_one_episode(learner, partner_pool, illegal_move_penalty=ILLEGAL_MOVE_PENALTY):
    """Plays one 2-player game: learner at seat 0, a randomly sampled
    partner at seat 1. Returns training data for the learner's OWN turns
    only, plus the episode score for logging.
    """
    partner_name = random.choice(list(partner_pool.keys()))
    partner = partner_pool[partner_name]
    if hasattr(partner, "reset"):
        try:
            partner.reset({})
        except TypeError:
            partner.reset()

    game_config = {"colors": 5, "ranks": 5, "players": 2,
                    "max_information_tokens": 8, "max_life_tokens": 3}
    env = rl_env_adjusted.make(obs_type="CHEAT", config=game_config)
    observations = env.reset()
    done = False

    learner_obs_vecs, learner_act_ints, learner_log_probs = [], [], []
    all_rewards = []  # every step's reward, both seats, for team reward-to-go
    learner_step_positions = []  # index into all_rewards where a learner action was taken
    illegal_count = 0

    while not done:
        obs0 = observations["player_observations"][0]
        obs1 = observations["player_observations"][1]
        current_player = obs0["current_player"]

        if current_player == 0:
            obs_vec, act_int, action, legal, log_prob = learner.act_int_and_legal(obs0)
            if legal:
                observations, rew, done, _ = env.step(action)
            else:
                illegal_count += 1
                rew = illegal_move_penalty
                # state doesn't advance on an illegal pick, matching the
                # convention used by the existing training scripts.
            learner_obs_vecs.append(obs_vec)
            learner_act_ints.append(act_int)
            learner_log_probs.append(log_prob)
            all_rewards.append(rew)
            learner_step_positions.append(len(all_rewards) - 1)
            if illegal_count > ILLEGAL_MOVE_MAX:
                break
        else:
            action = partner.act(obs1)
            observations, rew, done, _ = env.step(action)
            all_rewards.append(rew)

    ep_lives = observations["player_observations"][0]["life_tokens"]
    ep_firework = sum(observations["player_observations"][0]["fireworks"].values())
    ep_score = ep_firework if ep_lives > 0 else 0

    # Team reward-to-go over the WHOLE episode, then pick out just the
    # values at the learner's own decision points -- each of the learner's
    # actions gets credit for everything that happens afterward, including
    # points the partner scores off a hint the learner gave.
    rtg_full = basics.reward_to_go(all_rewards, gamma=GAMMA)
    learner_rtg = [rtg_full[i] for i in learner_step_positions]

    return (learner_obs_vecs, learner_act_ints, learner_log_probs, learner_rtg,
            ep_score, illegal_count, partner_name)


def train_one_epoch(learner, partner_pool, epoch_num):
    batch_obs, batch_log_probs, batch_rtg = [], [], []
    scores, illegal_counts = [], []
    steps_this_epoch = 0

    while steps_this_epoch < BATCH_SIZE:
        obs_vecs, act_ints, log_probs, rtg, score, illegal_count, partner_name = \
            run_one_episode(learner, partner_pool)
        batch_obs.extend(obs_vecs)
        batch_log_probs.extend(log_probs)
        batch_rtg.extend(rtg)
        scores.append(score)
        illegal_counts.append(illegal_count)
        steps_this_epoch += len(obs_vecs)

    rtg_t = torch.as_tensor(batch_rtg, dtype=torch.float32).to(learner.device)
    obs_t = torch.as_tensor(np.array(batch_obs), dtype=torch.float32).to(learner.device)

    # Value network: regress toward reward-to-go (the baseline target).
    value_preds = learner.value_network(obs_t).squeeze(-1)
    loss_vf = ((value_preds - rtg_t) ** 2).mean()
    learner.optimizer_vf.zero_grad()
    loss_vf.backward()
    learner.optimizer_vf.step()

    # Policy network: REINFORCE with the value network as a baseline
    # (advantage = reward-to-go - value estimate), detached so the policy
    # loss doesn't backprop into the value network.
    with torch.no_grad():
        baseline = learner.value_network(obs_t).squeeze(-1)
    advantage = rtg_t - baseline
    if advantage.std() > 1e-6:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    log_probs_t = torch.stack(batch_log_probs)
    loss_pi = -(log_probs_t * advantage).mean()
    learner.optimizer_pi.zero_grad()
    loss_pi.backward()
    learner.optimizer_pi.step()

    if epoch_num % SHOW_EPOCH_EVERY == 0:
        avg_score = sum(scores) / len(scores)
        avg_illegal = sum(illegal_counts) / len(illegal_counts)
        print(f"Epoch {epoch_num:5d} | episodes {len(scores):4d} | "
              f"avg score {avg_score:5.2f} | avg illegal/ep {avg_illegal:4.2f} | "
              f"loss_pi {loss_pi.item():7.4f} | loss_vf {loss_vf.item():7.4f}")

    return sum(scores) / len(scores)


def main():
    print(f"Training against partner pool: {PARTNER_NAMES}")
    partner_pool = build_partner_pool()
    print(f"Actually usable partners: {list(partner_pool.keys())}")

    learner = PopulationAgent()

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(learner, partner_pool, epoch)
        if SAVE_PARAMS_EVERY and epoch % SAVE_PARAMS_EVERY == 0:
            learner.save_policy(os.path.join(SAVE_DIR, f"policy_epoch{epoch}.pth"))

    final_path = os.path.join(SAVE_DIR, "policy_final.pth")
    learner.save_policy(final_path)
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
