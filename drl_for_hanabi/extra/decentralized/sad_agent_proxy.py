"""
SADAgentProxy: runs in the main process (torch 1.7), presents the same
Agent interface play_together.py expects (matching the pattern its
existing OBLAgentProxy hooks use), and forwards all real work to a
subprocess running under venv_sad (torch 2.4.1).

Deck synchronization strategy: unlike OBL, SAD's own copy of the HLE
engine has no deck-order control mechanism at all (see sad_agent.py's own
docstring), and unlike OBL, SAD's full deck is not known upfront (OBL
generates its entire 50-card deck before dealing; SAD's engine deals and
draws randomly, discovered only turn by turn). So SAD's shadow environment
is the "source of truth" for dealing/drawing, and this proxy mirrors what
it does into the main environment's own existing
reset_with_deck_order()/step_with_deck_order() machinery
(deck_sync_utils.py, reused unmodified): reset() reports the full initial
10-card deal for reset_with_deck_order() to consume all at once, and
apply_external_move() -- called once per turn, uniformly for every
player's move, including this agent's own, exactly matching
play_together.py's own calling convention for OBL -- appends any
newly-drawn replacement card onto the main environment's own
_synced_deck_order queue just before play_together.py's subsequent
step_with_deck_order() call consumes it, so both environments always draw
the identical card.

Dealing order, empirically verified (see test_sad_deck_sync.py):
sequential by player (all of player 0's hand_size cards, in slot order,
then all of player 1's), not interleaved round-robin.
"""
import json
import subprocess
import sys

COLOR_CHAR = ["R", "Y", "G", "W", "B"]

SAD_WORKER_PYTHON = "/home/percy/hanabi_sad_core/venv_sad/bin/python"
SAD_WORKER_SCRIPT = "/home/percy/hanabi_sad_core/sad_worker.py"
SAD_WORKER_CWD = "/home/percy/hanabi_sad_core"

DEFAULT_WEIGHT_FILE = "/home/percy/hanabi_SAD/models/op/sad-aux-op/M0.pthw"


def _card_tuple_to_deck_history_string(card):
    """(color_idx, rank_idx) -> deck_sync_utils' own string format, e.g.
    (4, 3) -> '4e' (rank digit = rank+1, color letter a-e = color_idx)."""
    color_idx, rank_idx = card
    rank_char = str(rank_idx + 1)
    color_char = chr(ord("a") + color_idx)
    return rank_char + color_char


class SADAgentProxy:
    def __init__(self, config=None):
        config = config or {}
        weight_file = config.get("weight_file", DEFAULT_WEIGHT_FILE)
        player_idx = config.get("player_idx", 0)
        num_players = config.get("num_players", 2)

        self.player_idx = player_idx
        self.num_players = num_players
        self._proc = subprocess.Popen(
            [SAD_WORKER_PYTHON, SAD_WORKER_SCRIPT],
            cwd=SAD_WORKER_CWD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        self._call({
            "cmd": "init",
            "weight_file": weight_file,
            "player_idx": player_idx,
            "num_players": num_players,
        })

    def _call(self, req):
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            resp = {"ok": False, "error": "Worker produced no output (may have crashed)."}
        else:
            resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(
                f"SAD worker error: {resp.get('error')}\n{resp.get('traceback', '')}"
            )
        return resp.get("result")

    def reset(self):
        """Resets SAD's own shadow environment and returns the initial
        10-card deal as a flat deck-order string list, in the exact
        sequential-by-player order reset_with_deck_order() expects --
        matching OBLAgentProxy's own deck_history()-based hook in
        play_together.py, but computed from a direct read-back of SAD's
        own dealt hands rather than from a deck_history() call (SAD's
        engine has no safe equivalent, and we deliberately never call the
        one it does have -- see sad_agent.py's own docstring)."""
        game_params = {
            "players": str(self.num_players), "colors": "5", "ranks": "5",
            "hand_size": "5", "max_information_tokens": "8",
            "max_life_tokens": "3", "seed": str(__import__("random").randint(1, 10**6)),
        }
        result = self._call({"cmd": "reset", "game_params": game_params})
        initial_hands = result["initial_hands"]

        hand_size = len(initial_hands[0])
        deck_order = []
        for player in range(self.num_players):
            for slot in range(hand_size):
                card = tuple(initial_hands[player][slot])
                deck_order.append(_card_tuple_to_deck_history_string(card))
        return deck_order

    def act(self, observation=None):
        """Returns this agent's own move (in this project's own move-dict
        format) if it's this agent's turn, else None -- matching
        play_together.py's own calling convention, which calls this every
        turn for every agent seated in the game. observation is accepted
        for interface compatibility but not used: SAD's own shadow
        environment independently tracks whose turn it is, and -- as long
        as sync is maintained -- always agrees with the main environment."""
        return self._call({"cmd": "act"})

    def apply_external_move(self, main_env, move_dict):
        """Called once per turn, for every player's move (including this
        agent's own, when it was this agent's turn) -- mirrors
        play_together.py's own calling convention for OBL, with one
        addition: since SAD's full deck isn't known upfront (see this
        module's own docstring), this also appends any card SAD's shadow
        environment drew as a result onto main_env's own synced-deck
        queue, immediately before play_together.py's subsequent
        step_with_deck_order() call consumes it.

        If this move ended the game (e.g. the last life token was just
        lost), no card is drawn or appended -- deck_sync_utils.py's own
        step_with_deck_order() checks is_terminal() before attempting to
        deal, so nothing further is expected here in that case."""
        result = self._call({"cmd": "apply_external_move", "move": move_dict})
        drawn = result["drawn_card"]
        if drawn is not None:
            main_env._synced_deck_order.append(
                _card_tuple_to_deck_history_string(tuple(drawn))
            )

    def close(self):
        if self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=5)
