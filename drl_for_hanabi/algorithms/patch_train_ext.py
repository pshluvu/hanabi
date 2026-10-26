import sys
path = sys.argv[1]
src = open(path).read()

def rep(old, new, count=1):
    global src
    assert src.count(old) == count, f"anchor count {src.count(old)} != {count}: {old[:60]!r}"
    src = src.replace(old, new)

rep('''class ExternalPartner:''',
'''def bombed(observations):
    return observations["player_observations"][0]["life_tokens"] == 0


class ExternalPartner:''')
rep('''"score": final_score(observations),
            "actions": actions}''',
    '''"score": final_score(observations),
            "bombed": bombed(observations), "actions": actions}''', count=2)

rep('''            scores, hints = [], np.zeros(10)''', '''            scores, hints, bombs = [], np.zeros(10), 0''')
rep('''                    scores.append(ep["score"])''', '''                    scores.append(ep["score"])
                    bombs += int(ep["bombed"])''')
rep('''f"   hints/game {hints.sum() / args.eval_episodes:5.2f}"''',
    '''f"   bombed {bombs:2d}/{args.eval_episodes}"
                  f"   hints/game {hints.sum() / args.eval_episodes:5.2f}"''')

rep('''    train_pool, holdout_pool = build_pool(hand_coded=args.partners)
    names = list(train_pool.keys())''',
'''    train_pool, holdout_pool = build_pool(hand_coded=args.partners)
    for n in (args.train_external or []):
        train_pool[n] = ExternalPartner(n)
        print(f"pool: external training partner '{n}' loaded (seat 0)")
    names = list(train_pool.keys())''')
rep('''            ep = play_vs_partner(learner, train_pool[name], seat=random.randint(0, 1),
                                 explore=True, type_id=type_id)
''', '''            try:
                ep = play_vs_partner(learner, train_pool[name], seat=random.randint(0, 1),
                                     explore=True, type_id=type_id)
            except Exception as e:
                if not isinstance(train_pool[name], ExternalPartner):
                    raise
                print(f"external partner '{name}' failed ({e}); restarting it", flush=True)
                train_pool[name].close()
                train_pool[name] = ExternalPartner(name)
                continue
''')
rep('''    save(learner, os.path.join(run_dir, "policy_final.pth"), meta)


def jensen_shannon''', '''    save(learner, os.path.join(run_dir, "policy_final.pth"), meta)
    for p in train_pool.values():
        if isinstance(p, ExternalPartner):
            p.close()


def jensen_shannon''')
rep('''    ap.add_argument("--checkpoint", default=None)''',
'''    ap.add_argument("--train-external", nargs="+", default=None, choices=["obl", "sad"],
                    help="population mode: also TRAIN against OBL and/or SAD")
    ap.add_argument("--checkpoint", default=None)''')
open(path, "w").write(src)
print("patched OK")
