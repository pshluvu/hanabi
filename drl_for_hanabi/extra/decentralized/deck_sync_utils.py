"""
Utilities for synchronizing your main repo's HLE environment to a specific,
predetermined deck order (e.g. one obtained from OBL's environment via
deck_history()), so both environments deal the same cards to the same
players in the same order -- a prerequisite for OBL and population to
play a genuinely shared, cooperative game.

No modification to the underlying C++ engine is required: this works
entirely through the existing apply_move() Python API, using DEAL-type
moves in place of the random dealing that deal_random_card() would
otherwise perform.
"""

from hanabi_learning_environment import pyhanabi

# Matches util.cc's ColorIndexToChar mapping (also used throughout this
# project, e.g. bram_best_rule_based_agent.py's COLORS list).
_COLOR_LETTERS = ['R', 'Y', 'G', 'W', 'B']


def obl_deck_string_to_color_rank(card_str):
    """Converts a single OBL deck_history() entry (e.g. '5e') into this
    repo's own (color_letter, rank_int) representation (e.g. ('B', 4)).

    OBL's encoding (hanabi_state.h's DeckHistory): rank is the internal
    0-indexed rank plus 1 (as a digit), and color is one of 'a'-'e'
    (index 0-4), NOT the 'RYGWB' letters used elsewhere in this project.
    """
    rank_char, color_char = card_str[0], card_str[1]
    rank = int(rank_char) - 1
    color_index = ord(color_char) - ord('a')
    color = _COLOR_LETTERS[color_index]
    return color, rank


def reset_with_deck_order(env, deck_order_strings):
    """Resets env (this project's HanabiEnv, from rl_env_adjusted.py) using
    a specific deck order instead of a random one.

    Args:
        env: an already-constructed HanabiEnv instance (same class used by
            rl_env_adjusted.make(...) throughout this project).
        deck_order_strings: list of strings in OBL's deck_history() format
            (e.g. ['5e', '3a', ...]), in the order cards should be dealt.

    Returns:
        The same observation dict format env.reset() normally returns.

    Also stores the deck order and a cursor on `env` itself (as
    `_synced_deck_order` / `_synced_deck_index`), so step_with_deck_order()
    can continue drawing replacement cards from the same predetermined
    sequence after this initial deal. Without this, only the INITIAL hands
    would match a synced shadow environment (e.g. OBL's own) -- every
    replacement card drawn after a PLAY/DISCARD would otherwise still come
    from this environment's own independent internal RNG, causing the two
    environments' hands to progressively diverge over the course of a game.
    """
    color_to_index = {c: i for i, c in enumerate(_COLOR_LETTERS)}
    env.state = env.game.new_initial_state()
    deck_iter = iter(deck_order_strings)
    dealt_count = 0
    while env.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        color, rank = obl_deck_string_to_color_rank(next(deck_iter))
        move = pyhanabi.HanabiMove.get_deal_move(color_to_index[color], rank)
        env.state.apply_move(move)
        dealt_count += 1

    env._synced_deck_order = list(deck_order_strings)
    env._synced_deck_index = dealt_count

    obs = env._make_observation_all_players()
    obs["current_player"] = env.state.cur_player()
    return obs


def step_with_deck_order(env, action):
    """Drop-in replacement for env.step(action) that deals any replacement
    cards (after a PLAY or DISCARD) from the same predetermined deck order
    reset_with_deck_order() established, instead of env's own internal RNG.

    Must only be called on an env that reset_with_deck_order() was already
    used on this episode (so env._synced_deck_order/_synced_deck_index
    exist). Mirrors rl_env_adjusted.py's own step() implementation exactly,
    except for how replacement chance moves get resolved.
    """
    color_to_index = {c: i for i, c in enumerate(_COLOR_LETTERS)}

    if isinstance(action, dict):
        action = env._build_move(action)
    elif isinstance(action, int):
        action = env.game.get_move(action)
    else:
        raise ValueError("Expected action as dict or int, got: {}".format(action))

    last_score = env.state.score()
    env.state.apply_move(action)

    deck_order = env._synced_deck_order
    # Stop dealing once the game has ended (e.g. the last life token was
    # just lost) even if cur_player() still reports a chance node --
    # otherwise, for integrations whose deck order is discovered
    # incrementally rather than known in full upfront, there may be no
    # further entries to deal, or dealing one anyway could be an
    # illegal move once the engine's own remaining-card counts are
    # exhausted.
    while env.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID and not env.state.is_terminal():
        card_str = deck_order[env._synced_deck_index]
        env._synced_deck_index += 1
        color, rank = obl_deck_string_to_color_rank(card_str)
        move = pyhanabi.HanabiMove.get_deal_move(color_to_index[color], rank)
        env.state.apply_move(move)

    observation = env._make_observation_all_players()
    done = env.state.is_terminal()
    reward = env.state.score() - last_score
    info = {}
    return (observation, reward, done, info)
