import sys
sys.path.insert(0, "/home/percy/hanabi/drl_for_hanabi/extra/decentralized")
from sad_agent_proxy import SADAgentProxy, _card_tuple_to_deck_history_string
from hanabi_learning_environment import rl_env_adjusted

def p(msg):
    print(msg, flush=True)

game_config = {
    "colors": 5, "ranks": 5, "players": 2, "hand_size": 5,
    "max_information_tokens": 8, "max_life_tokens": 3,
}
main_env = rl_env_adjusted.make(obs_type="CARD_KNOWLEDGE", config=game_config)

sad_game_params = {
    "players": "2", "colors": "5", "ranks": "5", "hand_size": "5",
    "max_information_tokens": "8", "max_life_tokens": "3", "seed": "1",
}

proxy = SADAgentProxy(player_idx=0, num_players=2)

# Reset SAD directly first (bypassing the proxy's reset, so we can inspect
# initial_hands before it gets consumed into a deck-order string).
result = proxy._call({"cmd": "reset", "game_params": sad_game_params})
initial_hands = result["initial_hands"]
p(f"SAD initial_hands (by player): {initial_hands}")

hand_size = len(initial_hands[0])
deck_order = []
for player in range(2):
    for slot in range(hand_size):
        card = tuple(initial_hands[player][slot])
        deck_order.append(_card_tuple_to_deck_history_string(card))
p(f"Reconstructed deck_order (round-robin guess): {deck_order}")

from deck_sync_utils import reset_with_deck_order
reset_with_deck_order(main_env, deck_order)

main_hands = main_env.state.player_hands()
p(f"Main env hands after sync: {[[str(c) for c in h] for h in main_hands]}")

# Manually format SAD hands the same way as main_env's card __str__ for comparison
COLOR_CHAR = ["R", "Y", "G", "W", "B"]
sad_hands_str = [[COLOR_CHAR[c] + str(r + 1) for (c, r) in hand] for hand in initial_hands]
p(f"SAD hands (formatted to compare): {sad_hands_str}")

match = main_hands and [[str(c) for c in h] for h in main_hands] == sad_hands_str
p(f"MATCH: {match}")

proxy.close()
