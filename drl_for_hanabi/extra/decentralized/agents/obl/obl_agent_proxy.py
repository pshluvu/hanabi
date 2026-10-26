"""
obl_agent_proxy.py -- runs in the MAIN project's own torch 1.7.0 process.
Looks and behaves like any other agent in agent_registry.py (same
Agent interface: reset(), act(observation), deck_history(),
apply_external_move()), but internally forwards every call to a
persistent subprocess (obl_worker.py) running under venv_obl's Python
(torch 2.4.1), where the actual OBL model inference happens.

This exists because the main project's torch 1.7.0 and OBL's compiled
hanalearn/rela modules (built against torch 2.4.1) cannot be imported
in the same Python process.
"""

import sys
import json
import subprocess

from hanabi_learning_environment.rl_env import Agent

OBL_WORKER_PYTHON = "/home/percy/off-belief-learning/venv_obl/bin/python"
OBL_WORKER_SCRIPT = "/home/percy/hanabi/drl_for_hanabi/extra/decentralized/agents/obl/obl_worker.py"


class OBLAgentProxy(Agent):
    """Configure with:
        {
            "weight_file": "/path/to/model0.pthw",
            "player_idx": 0,
            "num_players": 2,
        }
    """

    def __init__(self, config, *args, **kwargs):
        self.player_idx = config.get("player_idx", 0)
        self.num_players = config.get("num_players", 2)
        weight_file = config["weight_file"]

        self.process = subprocess.Popen(
            [
                OBL_WORKER_PYTHON, OBL_WORKER_SCRIPT,
                "--weight-file", weight_file,
                "--player-idx", str(self.player_idx),
                "--num-players", str(self.num_players),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        # Block until the (slow) model load inside the worker completes.
        # Skip any plain-text lines from libraries the worker imports
        # (e.g. off-belief-learning's own unflushed "loading file from: ..."
        # print) that land on stdout before our own JSON ready signal.
        ready = None
        while True:
            ready_line = self.process.stdout.readline()
            if not ready_line:
                stderr_output = self.process.stderr.read()
                raise RuntimeError(
                    f"OBL worker produced no ready signal (likely crashed on startup).\n"
                    f"Worker stderr:\n{stderr_output}")
            try:
                candidate = json.loads(ready_line)
            except json.JSONDecodeError:
                continue  # not JSON -- a library's own stray print, skip it
            if isinstance(candidate, dict) and candidate.get("ready"):
                ready = candidate
                break
        assert ready.get("ready"), f"OBL worker failed to start: {ready_line}"

    def _send(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        while True:
            response_line = self.process.stdout.readline()
            if not response_line:
                stderr_output = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(
                    f"OBL worker process died unexpectedly. Stderr: {stderr_output}")
            try:
                return json.loads(response_line)
            except json.JSONDecodeError:
                continue  # a library's own stray print landed on stdout, skip it

    def reset(self, config=None):
        self._send({"cmd": "reset"})

    def deck_history(self):
        return self._send({"cmd": "deck_history"})["deck"]

    def act(self, observation):
        request = {"cmd": "act", "has_obs": observation is not None}
        if observation is not None:
            request["current_player_offset"] = observation.get("current_player_offset", 0)
        return self._send(request)["action"]

    def apply_external_move(self, move_dict):
        self._send({"cmd": "apply_external_move", "move": move_dict})

    def shutdown(self):
        """Call when done with this agent (e.g. at the very end of a
        play_together.py run) to cleanly terminate the worker subprocess.
        Not calling this just leaves an orphaned process until the parent
        exits, so it's a courtesy, not a strict requirement.
        """
        try:
            self._send({"cmd": "shutdown"})
        except Exception:
            pass
        self.process.terminate()
