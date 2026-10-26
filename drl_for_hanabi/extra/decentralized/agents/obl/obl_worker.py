"""
obl_worker.py -- standalone subprocess worker that runs a real OBLAgent
inside venv_obl's Python (torch 2.4.1, matching hanalearn/rela's build),
so it can coexist with the main project's own torch 1.7.0 process.

Communicates over stdin/stdout using one JSON object per line. See
obl_agent_proxy.py (runs in the MAIN process) for the client side of
this protocol.

Usage (only ever invoked as a subprocess, not run directly):
    /home/percy/off-belief-learning/venv_obl/bin/python obl_worker.py \\
        --weight-file /path/to/model0.pthw --player-idx 0 --num-players 2
"""

import sys
import json
import argparse

sys.path.insert(0, "/home/percy/hanabi")
sys.path.insert(0, "/home/percy/hanabi/drl_for_hanabi/extra/decentralized/agents/obl")
from obl_agent import OBLAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-file", required=True)
    parser.add_argument("--player-idx", type=int, required=True)
    parser.add_argument("--num-players", type=int, required=True)
    args = parser.parse_args()

    agent = OBLAgent({
        "weight_file": args.weight_file,
        "player_idx": args.player_idx,
        "num_players": args.num_players,
    })

    # Signal readiness once the (slow) model load is done, before entering
    # the main loop -- the proxy waits for this line before sending commands.
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        cmd = request["cmd"]

        if cmd == "reset":
            agent.reset()
            response = {"ok": True}

        elif cmd == "deck_history":
            response = {"deck": agent.deck_history()}

        elif cmd == "act":
            # Only current_player_offset actually matters to OBLAgent.act();
            # everything else about the main project's observation format
            # is ignored, so we avoid shipping the whole (large, possibly
            # non-JSON-serializable) observation dict over the pipe.
            offset = request.get("current_player_offset", 0)
            fake_obs = {"current_player_offset": offset} if request.get("has_obs", True) else None
            action = agent.act(fake_obs)
            response = {"action": action}

        elif cmd == "apply_external_move":
            move_dict = request["move"]
            agent.apply_external_move(move_dict)
            response = {"ok": True}

        elif cmd == "shutdown":
            print(json.dumps({"ok": True}), flush=True)
            break

        else:
            response = {"error": f"unknown command: {cmd}"}

        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
