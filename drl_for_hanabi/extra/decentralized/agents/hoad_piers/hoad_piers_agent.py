"""
Wrapper for a HOAD imitator agent (imitation-learned from the Walton-Rivers
'piers' agent), making it compatible with this repo's Agent interface so it
can be registered in agent_registry.py and played against any other agent
via play_together.py.

This agent's neural network lives in a completely separate repo (HOAD,
github.com/aronsar/hoad), trained via imitator_agents/train.py on replay
data generated from the rule-based 'piers' agent. It expects the standard,
raw 658-dimensional HLE observation vector (obs['vectorized']) -- NOT the
custom reduced encodings (obs_to_vec62_discards, etc.) used by this repo's
own spg/vpg/ppo/population agents. Since both repos are built on the same
underlying DeepMind Hanabi Learning Environment, this raw vector is
available regardless of which repo produced the environment instance.
"""

import os
import sys
import numpy as np

from hanabi_learning_environment.rl_env import Agent

# Path to the cloned HOAD repo -- adjust if you cloned it somewhere else.
HOAD_ROOT = os.path.expanduser("~/hoad")
HOAD_IMITATOR_DIR = os.path.join(HOAD_ROOT, "imitator_agents")

# HOAD's own code needs to be importable to reconstruct the Mlp architecture.
if HOAD_IMITATOR_DIR not in sys.path:
    sys.path.insert(0, HOAD_IMITATOR_DIR)

from mlp import Mlp  # noqa: E402  (import must come after sys.path insert)
from tensorflow.keras.layers import ReLU  # noqa: E402
from tensorflow.keras.layers import Softmax  # noqa: E402

# Confirmed from cross_validation.py's `glb` class in the HOAD repo.
SIZE_OBS_VEC = 658
SIZE_ACT_VEC = 20

# Confirmed from train.py's hyperparameter dict in the HOAD repo.
HL_SIZES = [1024, 1024, 512, 512, 512, 256]
HL_ACTIVATIONS = [ReLU, ReLU, ReLU, ReLU, ReLU, ReLU]


class HOADImitatorAgent(Agent):
    """Loads a trained HOAD imitator model (e.g. saved_models/piers.save/best.h5)
    and exposes it through this repo's standard act(observation) interface.
    """

    def __init__(self, config=None):
        config = config or {}
        checkpoint_path = config.get(
            "checkpoint_path",
            os.path.join(HOAD_IMITATOR_DIR, "saved_models", "piers.save", "best.h5"),
        )

        # lr/batch_size/decay only matter for further training, not inference,
        # so placeholder values are fine here -- we only ever call predict().
        self._mlp = Mlp(
            io_sizes=(SIZE_OBS_VEC, SIZE_ACT_VEC),
            out_activation=Softmax,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            lr=0.001,
            batch_size=128,
            hl_activations=HL_ACTIVATIONS,
            hl_sizes=HL_SIZES,
            decay=0.0,
            bNorm=True,
            dropout=True,
            verbose=0,
        )
        self._mlp.construct_model(checkpoint_path, weights_only=True)
        print(f"HOADImitatorAgent loaded weights from: {checkpoint_path}")

    def act(self, observation):
        if observation.get("current_player_offset", None) != 0:
            return None

        obs_vec = np.array(observation["vectorized"], dtype=np.float32).reshape(1, -1)
        action_probs = self._mlp.model.predict(obs_vec, verbose=0)[0]

        legal_moves_as_int = observation["legal_moves_as_int"]
        legal_moves = observation["legal_moves"]

        # Mask illegal actions by zeroing their probability, then pick the
        # legal action with the highest remaining probability. Using the
        # parallel legal_moves / legal_moves_as_int lists (rather than
        # hand-decoding the action-index convention ourselves) avoids any
        # risk of getting HOAD's specific int-encoding scheme wrong.
        best_idx_in_legal_list = None
        best_prob = -1.0
        for i, action_int in enumerate(legal_moves_as_int):
            if action_probs[action_int] > best_prob:
                best_prob = action_probs[action_int]
                best_idx_in_legal_list = i

        return legal_moves[best_idx_in_legal_list]
