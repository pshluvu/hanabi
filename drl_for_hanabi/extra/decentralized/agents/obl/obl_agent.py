"""
OBLAgent: wraps a pretrained Off-Belief Learning (OBL) model, from the
off-belief-learning repo, to play within this project's own
play_together.py / agent_registry.py framework alongside population,
bram_best, and the other agents.

Since OBL's model was trained using a differently-encoded environment
than this project's own (a custom, 533/658-dim observation format,
computed inside the off-belief-learning repo's own compiled hanalearn
module), this wrapper maintains its own "shadow" instance of OBL's
environment, kept in lockstep with the main environment:
  - at the start of each episode, both environments are dealt the exact
    same deck (see deck_sync_utils.py for the main-environment side of
    this; OBL's own environment IS the deck's source of truth here)
  - every move taken by ANY agent in the main game (including this one)
    is mirrored into this shadow environment via apply_external_move(),
    so OBL's belief state always matches the real game state

This lets OBL make decisions using its own genuine observation encoding
and trained policy, while still participating in a single, shared,
cooperative game with population and the project's other agents.
"""

import sys
import torch

OBL_REPO_PATH = "/home/percy/off-belief-learning"
sys.path.insert(0, f"{OBL_REPO_PATH}/build")
sys.path.insert(0, f"{OBL_REPO_PATH}/pyhanabi")

import rela  # noqa: E402  (must be imported before hanalearn)
import hanalearn  # noqa: E402
import utils as obl_utils  # noqa: E402

from hanabi_learning_environment.rl_env import Agent  # noqa: E402

_COLOR_LETTERS = ['R', 'Y', 'G', 'W', 'B']
_COLOR_TO_INDEX = {c: i for i, c in enumerate(_COLOR_LETTERS)}


def obl_deck_string_to_color_rank(card_str):
    """Converts a deck_history()-format entry (e.g. '5e') into
    (color_index, rank). Matches deck_sync_utils.py's identical helper
    exactly -- duplicated here rather than imported, since this module
    runs inside the OBL worker subprocess, whose sys.path doesn't include
    the main repo's extra/decentralized/ directory where
    deck_sync_utils.py lives.
    """
    rank_char, color_char = card_str[0], card_str[1]
    rank = int(rank_char) - 1
    color_index = ord(color_char) - ord('a')
    return color_index, rank


def _generate_random_deck_order():
    """Generates a random, correctly-shuffled 50-card deck order, in the
    same string format the (unsafe) deck_history() C++ method would have
    produced (e.g. '5e' = rank 4 (displayed 5), color index 4).

    Used instead of ever calling deck_history(), which has a confirmed
    destructive side effect on OBL's own C++ engine: calling it silently
    corrupts other, separately-constructed HanabiEnv/HanabiState
    instances too, not just the one it's called on (confirmed via direct
    C++ debug instrumentation).
    """
    import random
    rank_counts = {0: 3, 1: 2, 2: 2, 3: 2, 4: 1}
    deck = []
    for color_index in range(5):
        color_char = chr(ord('a') + color_index)
        for rank, count in rank_counts.items():
            rank_char = str(rank + 1)
            for _ in range(count):
                deck.append(rank_char + color_char)
    random.shuffle(deck)
    return deck


class OBLAgent(Agent):
    """Plays using a pretrained OBL model. Configure with:
        {
            "weight_file": "/path/to/model0.pthw",
            "player_idx": 0,  # this agent's own seat at the table
            "num_players": 2,
        }
    """

    def __init__(self, config, *args, **kwargs):
        self.player_idx = config.get("player_idx", 0)
        self.num_players = config.get("num_players", 2)
        weight_file = config["weight_file"]

        overwrite = {"vdn": False, "device": "cpu", "boltzmann_act": False}
        self.agent, self.cfg = obl_utils.load_agent(weight_file, overwrite)
        self.agent.train(False)
        self.use_sad_obs = bool(self.cfg.get("sad", False))

        self.obl_env = None
        self.hidden = None

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, config=None):
        """Starts this agent's own shadow environment fresh. Also tracks
        this episode's full deck order internally, so apply_external_move()
        can manually resolve replacement-card draws after every PLAY/
        DISCARD -- OBL's own HanabiEnv.step() does NOT do this
        automatically (unlike this project's own rl_env_adjusted.py
        step()), so without this, hands would never refill at all.

        IMPORTANT: deck_history() has a destructive side effect -- it
        drains the remaining deck as part of reconstructing the history
        list (confirmed via direct C++ debug instrumentation; matches a
        warning comment in the original source about this function
        potentially being unsafe after gameplay has started). We must
        therefore sample the deck order from a separate, throwaway
        environment, never from the one we intend to actually keep
        playing on.
        """
        game_params = {
            "players": str(self.num_players),
            "colors": "5",
            "ranks": "5",
            "hand_size": "5",
            "max_information_tokens": "8",
            "max_life_tokens": "3",
            "seed": "1",
        }
        deck = _generate_random_deck_order()
        # OBL's own internal deck_order_ is consumed back-to-front (via
        # .back()/.pop_back()), while this project's own step_with_deck_order()
        # consumes the same shared list front-to-back. Reversing here (only
        # for OBL's own internal list) makes both sides draw the same cards
        # in the same actual order, without needing to change the shared
        # deck_history()-format list itself (which the main environment
        # sync still expects in original, front-to-back order).
        deck_values = [
            hanalearn.HanabiCardValue(*obl_deck_string_to_color_rank(card_str))
            for card_str in reversed(deck)
        ]
        self.obl_env = hanalearn.HanabiEnv(game_params, -1, False)
        self.obl_env.reset_with_deck(deck_values)
        self.hidden = None
        self._deck_order = deck

    def deck_history(self):
        """This agent's own tracked deck order, in the same string format
        deck_history() would have produced (e.g. '5e'). Use this as the
        source of truth to sync the main environment via
        deck_sync_utils.reset_with_deck_order().

        Returns our own Python-generated deck list directly -- never calls
        the real deck_history() C++ method, which has a confirmed
        destructive side effect that silently corrupts other, separately-
        constructed HanabiEnv/HanabiState instances too.
        """
        return self._deck_order

    # ------------------------------------------------------------------
    # Turn-by-turn play
    # ------------------------------------------------------------------

    def act(self, observation):
        """Follows this project's standard Agent convention: returns None
        unless it is this agent's own turn (observation['current_player_offset']
        == 0), matching every other agent in this project. When it IS this
        agent's turn, the passed-in `observation` itself is otherwise
        ignored -- this agent observes via its own shadow environment
        instead, which has been kept in sync via apply_external_move().
        Returns an action in this project's own dict format, e.g.
        {'action_type': 'PLAY', 'card_index': 2}.
        """
        if observation is not None and observation.get('current_player_offset', 0) != 0:
            return None

        state = self.obl_env.get_hle_state()
        if self.use_sad_obs:
            obs = hanalearn.observe_sad(state, self.player_idx)
        else:
            obs = hanalearn.observe(state, self.player_idx, False)

        if self.hidden is None:
            self.hidden = self.agent.get_h0(1)
            # reshape from net's own [num_layer, batch, dim] to the public
            # act() interface's expected [batch, num_layer, num_player, dim]
            self.hidden = {
                k: v.permute(1, 0, 2).unsqueeze(2).contiguous()
                for k, v in self.hidden.items()
            }

        full_obs = {
            "priv_s": obs["priv_s"].unsqueeze(0),
            "publ_s": obs["publ_s"].unsqueeze(0),
            "legal_move": obs["legal_move"].unsqueeze(0),
            **self.hidden,
        }
        result = self.agent.act(full_obs)
        action_idx = result["a"].item()
        # keep new hidden state in the same [batch, num_layer, num_player, dim]
        # shape this method always stores it in
        self.hidden = {
            "h0": result["h0"],
            "c0": result["c0"],
        }

        obl_move = self.obl_env.get_move(action_idx)
        return self._obl_move_to_project_dict(obl_move)

    def apply_external_move(self, move_dict):
        """Call this after EVERY move in the main game (from any agent,
        including this one), so this agent's shadow environment stays
        synchronized with the real game state.

        Just calls step() -- OBL's own HanabiEnv.step() automatically
        resolves any resulting chance node (replacement card draw) using
        its own internal deck_order_ (set via reset_with_deck() during
        reset()), so no manual chance resolution is needed here.
        """
        obl_move = self._project_dict_to_obl_move(move_dict)
        self.obl_env.step(obl_move)

    # ------------------------------------------------------------------
    # Move format translation
    # ------------------------------------------------------------------

    def _obl_move_to_project_dict(self, obl_move):
        move_type = obl_move.move_type()
        if move_type == hanalearn.MoveType.Play:
            return {'action_type': 'PLAY', 'card_index': obl_move.card_index()}
        elif move_type == hanalearn.MoveType.Discard:
            return {'action_type': 'DISCARD', 'card_index': obl_move.card_index()}
        elif move_type == hanalearn.MoveType.RevealColor:
            return {
                'action_type': 'REVEAL_COLOR',
                'target_offset': obl_move.target_offset(),
                'color': _COLOR_LETTERS[obl_move.color()],
            }
        elif move_type == hanalearn.MoveType.RevealRank:
            return {
                'action_type': 'REVEAL_RANK',
                'target_offset': obl_move.target_offset(),
                'rank': obl_move.rank(),
            }
        else:
            raise ValueError(f"Unsupported OBL move type: {move_type}")

    def _project_dict_to_obl_move(self, move_dict):
        if isinstance(move_dict, int):
            # Some agents in this project (e.g. Population_Dec_Agent) return
            # a raw global move UID rather than a formatted dict -- the same
            # UID-to-move mapping used by rl_env_adjusted.step() (via
            # self.game.get_move(action)) is available here as
            # self.obl_env.get_move(), since both projects are built on the
            # same underlying HLE engine with matching game parameters.
            return self.obl_env.get_move(move_dict)

        action_type = move_dict['action_type']
        if action_type == 'PLAY':
            return hanalearn.HanabiMove(
                hanalearn.MoveType.Play, move_dict['card_index'], -1, -1, -1)
        elif action_type == 'DISCARD':
            return hanalearn.HanabiMove(
                hanalearn.MoveType.Discard, move_dict['card_index'], -1, -1, -1)
        elif action_type == 'REVEAL_COLOR':
            return hanalearn.HanabiMove(
                hanalearn.MoveType.RevealColor, -1, move_dict['target_offset'],
                _COLOR_TO_INDEX[move_dict['color']], -1)
        elif action_type == 'REVEAL_RANK':
            return hanalearn.HanabiMove(
                hanalearn.MoveType.RevealRank, -1, move_dict['target_offset'],
                -1, move_dict['rank'])
        else:
            raise ValueError(f"Unsupported project move type: {action_type}")
