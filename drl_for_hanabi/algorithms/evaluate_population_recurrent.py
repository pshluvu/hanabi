"""
Standalone evaluation for the recurrent population agent.

population_recurrent isn't wired into agent_registry.py / play_together.py
(that registry's "population" entry points at the MLP's policy_final.pth,
and wiring in a second, differently-shaped network there would mean writing
a whole new wrapper agent class). This script sidesteps that: it loads
policy_final.pth from population_training_recurrent.py directly, and plays
it -- greedily, no exploration, no training -- against a chosen partner for
many episodes, the same way play_together.py's own evaluation works.

Usage:
    cd drl_for_hanabi/algorithms
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
    TF_ENABLE_ONEDNN_OPTS=0 KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
    python evaluate_population_recurrent.py --partner bram_best --episodes 50
    python evaluate_population_recurrent.py --partner self --episodes 50
"""
import sys
import os
import argparse

_ALGO_DIR = os.path.dirname(os.path.realpath(__file__))
_DRL_DIR = os.path.dirname(_ALGO_DIR)
_REPO_ROOT = os.path.dirname(_DRL_DIR)
_DECENTRALIZED_DIR = os.path.join(_DRL_DIR, "extra", "decentralized")
for _p in (_REPO_ROOT, _DECENTRALIZED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False

from hanabi_learning_environment import rl_env_adjusted
from agent_registry import build_agent

# Import the agent class + checkpoint path from the training script itself,
# so this evaluation always matches whatever architecture actually trained.
from population_training_recurrent import PopulationAgentRecurrent, SAVE_DIR


def learner_act_greedy(learner, obs, h_pi, h_vf):
    """Greedy (explore=False) version of act_int_and_legal -- picks the
    highest-probability action instead of sampling, which is standard for
    EVALUATING a policy (vs. training, which needs exploration)."""
    (act_int, action, legal, log_prob, entropy, value_pred,
     h_pi_next, h_vf_next) = learner.act_int_and_legal(obs, h_pi, h_vf, explore=False)
    return action, legal, h_pi_next, h_vf_next


def play_one_episode(learner, partner, learner_is_recurrent_self=False, h_pi_partner=None, h_vf_partner=None):
    game_config = {"colors": 5, "ranks": 5, "players": 2,
                   "max_information_tokens": 8, "max_life_tokens": 3}
    env = rl_env_adjusted.make(obs_type="CARD_KNOWLEDGE", config=game_config)
    observations = env.reset()
    done = False

    h_pi, h_vf = learner.new_episode_hidden()
    if learner_is_recurrent_self:
        h_pi_partner, h_vf_partner = learner.new_episode_hidden()

    illegal_count = 0
    ILLEGAL_MOVE_MAX = 50

    while not done:
        obs0 = observations["player_observations"][0]
        obs1 = observations["player_observations"][1]
        current_player = obs0["current_player"]

        if current_player == 0:
            action, legal, h_pi, h_vf = learner_act_greedy(learner, obs0, h_pi, h_vf)
            if legal:
                observations, rew, done, _ = env.step(action)
            else:
                illegal_count += 1
                if illegal_count > ILLEGAL_MOVE_MAX:
                    break
        else:
            if learner_is_recurrent_self:
                action, legal, h_pi_partner, h_vf_partner = learner_act_greedy(
                    learner, obs1, h_pi_partner, h_vf_partner)
                if legal:
                    observations, rew, done, _ = env.step(action)
                else:
                    illegal_count += 1
                    if illegal_count > ILLEGAL_MOVE_MAX:
                        break
            else:
                action = partner.act(obs1)
                observations, rew, done, _ = env.step(action)

    ep_lives = observations["player_observations"][0]["life_tokens"]
    ep_firework = sum(observations["player_observations"][0]["fireworks"].values())
    ep_score = ep_firework if ep_lives > 0 else 0
    return ep_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partner", type=str, required=True,
                         help="Agent name from agent_registry.py (e.g. bram_best), or 'self' for self-play")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    learner = PopulationAgentRecurrent()
    final_path = os.path.join(SAVE_DIR, "policy_final.pth")
    learner.load_policy(final_path)
    learner.policy_net.eval()
    learner.value_net.eval()

    self_play = (args.partner == "self")
    partner = None if self_play else build_agent(args.partner)

    scores = []
    with torch.no_grad():
        for ep in range(args.episodes):
            score = play_one_episode(learner, partner, learner_is_recurrent_self=self_play)
            scores.append(score)
            print(f"Episode {ep:3d} | score {score}")

    avg = sum(scores) / len(scores)
    print()
    print(f"Partner: {args.partner}")
    print(f"Episodes: {len(scores)}")
    print(f"Max score: {max(scores)}")
    print(f"Average score: {avg:.3f}")


if __name__ == "__main__":
    main()
