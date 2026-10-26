"""
Sanity check for the "cheat" agents (ppo / vpg / spg).

WHY: An agent that can see its own cards has no need to coordinate -- it can
just play its own playable cards -- so it should score well even next to an
unfamiliar partner. The proposal reports these agents scoring 19-22/25 in
self-play but 0/25 with rule-based partners. The examiners pointed out that
this looks like a policy/implementation problem, not a convention problem.
This script tries to tell which it is.

Two parts:

PART 1 -- what does the observation actually contain?
  For each observation type (CARD_KNOWLEDGE = standard rules, CHEAT = own hand
  visible) it prints whether the agent's OWN hand identities appear in the
  observation dict, and whether the 136-dim vector that the agents are fed
  changes when those identities change. If the vector does NOT depend on
  the own-hand identities, then "cheat" agents never actually see their
  cards, and the examiners' premise doesn't hold for the policy.

PART 2 -- how do the agents behave?
  Each agent plays N games against each partner, alternating seats. It
  reports, per pairing:
    score   : mean score under the repo's rule (0 if all lives are lost)
    fw      : mean fireworks BEFORE that zeroing (separates "scored
              nothing" from "bombed out after scoring something")
    bomb%   : fraction of games that ended with 0 lives
    misAG / misPT : mean misplays (lives lost) caused by the agent / partner
  A 0 caused by the agent's own misplays (misAG high) while it can see its
  cards is a policy-learning problem.

Usage (from drl_for_hanabi/algorithms):
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
    TF_ENABLE_ONEDNN_OPTS=0 KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
    python sanity_check_cheat_agents.py --episodes 30
"""
import sys
import os
import copy
import argparse

_ALGO_DIR = os.path.dirname(os.path.realpath(__file__))
_DRL_DIR = os.path.dirname(_ALGO_DIR)
_REPO_ROOT = os.path.dirname(_DRL_DIR)
_DECENTRALIZED_DIR = os.path.join(_DRL_DIR, "extra", "decentralized")
for _p in (_REPO_ROOT, _DECENTRALIZED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from hanabi_learning_environment import rl_env_adjusted
from drl_for_hanabi.utils import state_representation_dec
from agent_registry import build_agent

GAME_CONFIG = {"colors": 5, "ranks": 5, "players": 2,
               "max_information_tokens": 8, "max_life_tokens": 3}


def make_env(obs_type):
    return rl_env_adjusted.make(obs_type=obs_type, config=GAME_CONFIG)


# ----------------------------------------------------------------------------
# PART 1: what is in the observation?
# ----------------------------------------------------------------------------
def card_is_visible(card):
    """A card dict is 'visible' if its colour/rank are actually filled in."""
    color = card.get("color")
    rank = card.get("rank")
    return (color is not None) and (rank is not None) and (rank != -1)


def inspect_observation(obs_type):
    print(f"\n--- obs_type = {obs_type} ---")
    try:
        env = make_env(obs_type)
        observations = env.reset()
    except Exception as e:  # noqa: BLE001
        print(f"  could not build env with this obs_type: {e}")
        return
    obs0 = observations["player_observations"][0]
    print(f"  observation keys: {sorted(obs0.keys())}")

    hands = obs0.get("observed_hands")
    if hands is None:
        print("  no 'observed_hands' key -- cannot check own-hand visibility directly")
        return
    own = hands[0]
    partner = hands[1] if len(hands) > 1 else []
    print(f"  own hand (as the agent sees it):     {own}")
    print(f"  partner hand (as the agent sees it): {partner}")
    own_visible = all(card_is_visible(c) for c in own)
    print(f"  own hand identities visible in observation: {own_visible}")

    # Does the 136-dim vector change if own-hand identities change?
    base_vec = np.array(state_representation_dec.obs_to_vec136_binary(obs0))
    print(f"  encoded vector length: {len(base_vec)}, non-zero entries: {int(np.count_nonzero(base_vec))}")
    if not own_visible:
        print("  -> own-hand identities are NOT in the observation, so the encoder cannot depend on them.")
        return
    colors = ["R", "Y", "G", "W", "B"]
    changed_any = False
    for trial in range(5):
        mod = dict(obs0); mod["observed_hands"] = copy.deepcopy(obs0["observed_hands"])
        for card in mod["observed_hands"][0]:
            new_color = colors[(colors.index(card["color"]) + 1 + trial) % 5] if card["color"] in colors else card["color"]
            card["color"] = new_color
            card["rank"] = (card["rank"] + 1 + trial) % 5
        vec = np.array(state_representation_dec.obs_to_vec136_binary(mod))
        if not np.array_equal(vec, base_vec):
            changed_any = True
            break
    print(f"  encoded vector depends on own-hand identities: {changed_any}")
    if not changed_any:
        print("  -> WARNING: own cards are visible in the observation dict, but the 136-dim vector")
        print("     ignores them. 'Cheat' agents fed this vector effectively CANNOT see their cards.")


# ----------------------------------------------------------------------------
# PART 2: behaviour of the agents
# ----------------------------------------------------------------------------
def play_game(obs_type, agent_a, agent_b, agent_seat):
    """agent_a sits at agent_seat, agent_b at the other seat.
    Returns (score, raw_fireworks, lives_left, misplays_by_a, misplays_by_b)."""
    env = make_env(obs_type)
    observations = env.reset()
    seats = {agent_seat: agent_a, 1 - agent_seat: agent_b}
    done = False
    mis = {agent_seat: 0, 1 - agent_seat: 0}
    steps = 0
    while not done and steps < 500:
        steps += 1
        pobs = observations["player_observations"]
        cur = pobs[0]["current_player"]
        lives_before = pobs[0]["life_tokens"]
        action = seats[cur].act(pobs[cur])
        observations, _, done, _ = env.step(action)
        lives_after = observations["player_observations"][0]["life_tokens"]
        if lives_after < lives_before:
            mis[cur] += lives_before - lives_after
    final = observations["player_observations"][0]
    lives = final["life_tokens"]
    fw = sum(final["fireworks"].values())
    score = fw if lives > 0 else 0
    return score, fw, lives, mis[agent_seat], mis[1 - agent_seat]


def evaluate_pair(obs_type, agent_name, partner_name, episodes):
    agent = build_agent(agent_name)
    partner = build_agent(agent_name if partner_name == "self" else partner_name)
    rows = []
    for ep in range(episodes):
        seat = ep % 2
        try:
            rows.append(play_game(obs_type, agent, partner, seat))
        except Exception as e:  # noqa: BLE001
            print(f"    game {ep} failed ({type(e).__name__}: {e}); skipping")
    if not rows:
        return None
    arr = np.array(rows, dtype=float)
    return {
        "n": len(rows),
        "score": arr[:, 0].mean(),
        "fw": arr[:, 1].mean(),
        "bomb": float((arr[:, 2] <= 0).mean()),
        "misAG": arr[:, 3].mean(),
        "misPT": arr[:, 4].mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--agents", nargs="+", default=["ppo", "vpg", "spg"])
    parser.add_argument("--partners", nargs="+",
                        default=["self", "simple", "bram_simple", "bram_best"])
    parser.add_argument("--obs-types", nargs="+", default=["CARD_KNOWLEDGE", "CHEAT"])
    parser.add_argument("--skip-inspect", action="store_true")
    args = parser.parse_args()

    if not args.skip_inspect:
        print("=" * 72)
        print("PART 1: what do the observations contain?")
        print("=" * 72)
        for ot in args.obs_types:
            inspect_observation(ot)

    print()
    print("=" * 72)
    print(f"PART 2: agent behaviour ({args.episodes} games per cell, seats alternate)")
    print("=" * 72)
    for ot in args.obs_types:
        print(f"\n##### obs_type = {ot} #####")
        print(f"{'agent':<6} {'partner':<12} {'n':>3} {'score':>6} {'fw':>6} {'bomb%':>6} {'misAG':>6} {'misPT':>6}")
        for agent_name in args.agents:
            for partner_name in args.partners:
                try:
                    res = evaluate_pair(ot, agent_name, partner_name, args.episodes)
                except Exception as e:  # noqa: BLE001
                    print(f"{agent_name:<6} {partner_name:<12} FAILED: {type(e).__name__}: {e}")
                    continue
                if res is None:
                    print(f"{agent_name:<6} {partner_name:<12} no games completed")
                    continue
                print(f"{agent_name:<6} {partner_name:<12} {res['n']:>3} {res['score']:>6.2f} "
                      f"{res['fw']:>6.2f} {100*res['bomb']:>5.0f}% {res['misAG']:>6.2f} {res['misPT']:>6.2f}")
    print("\nHow to read this: a cheat agent that bombs out (high bomb%, high misAG) while")
    print("it can see its own cards points to a policy/implementation problem. High misPT")
    print("means the PARTNER is causing the zeros, which is a compatibility issue instead.")


if __name__ == "__main__":
    main()
