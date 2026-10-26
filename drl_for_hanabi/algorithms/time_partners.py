
import sys, time, torch, numpy as np
sys.argv = ["x"]
import population_training_v2 as P
from agent_registry import build_agent

ck = torch.load(P.PARAM_DIR + "/base_hand/policy_final.pth", map_location="cpu")
L = P.Learner(P.detect_obs_dim(), n_types=ck["n_types"], oracle=ck.get("oracle", False))
L.load_state(ck)
partners = {n: build_agent(n) for n in ["bram_simple", "bram_best", "hoad_piers"]}
partners["sad"] = P.ExternalPartner("sad")
partners["obl"] = P.ExternalPartner("obl")
with torch.no_grad():
    for n, p in partners.items():
        t = time.time(); sc = []
        for g in range(5):
            sc.append(P.play_vs_partner(L, p, seat=g % 2, explore=True)["score"])
        print(f"{n:12s} {(time.time() - t) / 5:6.2f} s/game   scores {sc}", flush=True)
