import sys, numpy as np
sys.argv = ["x"]
import population_training_v2 as P
from agent_registry import build_agent

class Stub:
    """Stands in for the learner: bram_simple plays through the OBL/SAD adapter."""
    def __init__(self): self.a = build_agent("bram_simple")
    def new_hidden(self): return None, None
    def act(self, obs, h_pi, h_vf, explore=False, type_id=None):
        m = self.a.act(obs)
        return {"idx": 0, "move": m, "log_prob": None, "entropy": None,
                "value": None, "aux": None, "h_pi": None, "h_vf": None}

for name in ["obl", "sad"]:
    ext = P.ExternalPartner(name)
    s = [P.play_vs_external(Stub(), ext, explore=False)["score"] for _ in range(15)]
    print(name, "with bram_simple through adapter:", np.mean(s), s, flush=True)
    ext.close()
