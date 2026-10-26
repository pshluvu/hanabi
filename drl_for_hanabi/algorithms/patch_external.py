import sys
path = sys.argv[1]
src = open(path).read()

def rep(old, new):
    global src
    assert src.count(old) == 1, f"anchor not found exactly once: {old[:60]!r}"
    src = src.replace(old, new)

ADAPTER = '''
class ExternalPartner:
    """OBL or SAD via the repo's subprocess proxies. Partner always sits in seat 0."""
    def __init__(self, name):
        from agent_registry import build_agent
        self.name = name
        self.agent = build_agent(name)

    def close(self):
        try:
            if self.name == "obl":
                self.agent.shutdown()
            else:
                self.agent.close()
        except Exception:
            pass


def move_to_dict(obs, move):
    if isinstance(move, dict):
        return move
    return obs["legal_moves"][obs["legal_moves_as_int"].index(move)]


def play_vs_external(learner, partner, explore, type_id=None):
    from deck_sync_utils import reset_with_deck_order, step_with_deck_order
    env = make_env()
    env.reset()
    ag, pseat, seat = partner.agent, 0, 1
    if partner.name == "obl":
        ag.reset()
        observations = reset_with_deck_order(env, ag.deck_history())
    else:
        observations = reset_with_deck_order(env, ag.reset())
    h_pi, h_vf = learner.new_hidden()
    rewards, steps, actions = [], [], []
    done, guard = False, 0
    while not done and guard < 500:
        guard += 1
        cur = observations["player_observations"][0]["current_player"]
        p_move = ag.act(observations["player_observations"][pseat])
        if cur == pseat:
            assert p_move is not None, "external partner returned no move on its own turn"
            move = p_move
        else:
            obs = observations["player_observations"][seat]
            out = learner.act(obs, h_pi, h_vf, explore=explore, type_id=type_id)
            h_pi, h_vf = out["h_pi"], out["h_vf"]
            move = move_to_dict(obs, out["move"])
            steps.append({"log_prob": out["log_prob"], "entropy": out["entropy"],
                          "value": out["value"], "aux": out["aux"], "pos": len(rewards)})
            actions.append(out["idx"])
        if partner.name == "obl":
            ag.apply_external_move(move)
        else:
            ag.apply_external_move(env, move)
        observations, rew, done, _ = step_with_deck_order(env, move)
        rewards.append(rew)
    return {"steps": steps, "rewards": rewards, "score": final_score(observations),
            "actions": actions}


def play_vs_partner(learner, partner, seat, explore, type_id=None, record=False):
    """One game, learner in `seat`, partner in the other. Returns record dict."""
    if isinstance(partner, ExternalPartner):
        return play_vs_external(learner, partner, explore, type_id)
'''
rep('''
def play_vs_partner(learner, partner, seat, explore, type_id=None, record=False):
    """One game, learner in `seat`, partner in the other. Returns record dict."""
''', ADAPTER)

rep('''    rows, hint_dists = [], {}
    for group, pool in (("train", train_pool), ("HELD-OUT", holdout_pool)):''',
'''    rows, hint_dists = [], {}
    ext_pool = {}
    for n in (args.external or []):
        try:
            ext_pool[n] = ExternalPartner(n)
            print(f"pool: external partner '{n}' loaded (seat 0)")
        except Exception as e:
            print(f"pool: SKIPPED external '{n}' ({e})")
    for group, pool in (("train", train_pool), ("HELD-OUT", holdout_pool), ("EXTERNAL", ext_pool)):''')

rep('''    held = [r[2] for r in rows if r[0] in ("HELD-OUT", "UNSEEN")]''',
'''    held = [r[2] for r in rows if r[0] in ("HELD-OUT", "UNSEEN")]
    ext = [r[2] for r in rows if r[0] == "EXTERNAL"]
    for p in ext_pool.values():
        p.close()''')
rep('''        print(f"mean over HELD-OUT partners (zero-shot): {np.mean(held):.2f}")''',
'''        print(f"mean over HELD-OUT partners (zero-shot): {np.mean(held):.2f}")
    if ext:
        print(f"mean over EXTERNAL partners (zero-shot): {np.mean(ext):.2f}")''')

rep('''    ap.add_argument("--checkpoint", default=None)''',
'''    ap.add_argument("--external", nargs="+", default=None, choices=["obl", "sad"],
                    help="eval only: also play against OBL and/or SAD (zero-shot)")
    ap.add_argument("--checkpoint", default=None)''')
open(path, "w").write(src)
print("patched OK")
