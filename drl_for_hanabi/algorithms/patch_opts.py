import sys
path = sys.argv[1]
src = open(path).read()

def rep(old, new, count=1):
    global src
    assert src.count(old) == count, f"anchor count {src.count(old)} != {count}: {old[:60]!r}"
    src = src.replace(old, new)

rep('''SELFPLAY_TRAIN_SEEDS = [0, 1, 2, 3]''', '''SKIP_SELFPLAY = False            # set by --no-selfplay
SELFPLAY_TRAIN_SEEDS = [0, 1, 2, 3]''')
rep('''    for target, seeds in ((train, SELFPLAY_TRAIN_SEEDS), (holdout, SELFPLAY_HOLDOUT_SEEDS)):''',
    '''    for target, seeds in (() if SKIP_SELFPLAY else
                          ((train, SELFPLAY_TRAIN_SEEDS), (holdout, SELFPLAY_HOLDOUT_SEEDS))):''')
rep('''    path, start = latest_checkpoint(run_dir)
    if path:
        ck = torch.load(path, map_location="cpu")
        if ck.get("partner_names") != names:''',
'''    path, start = latest_checkpoint(run_dir)
    if not path and args.init_from:
        ck0 = torch.load(args.init_from, map_location="cpu")
        learner.policy_net.load_state_dict(ck0["policy_net"])
        learner.value_net.load_state_dict(ck0["value_net"])
        print(f"initialised policy/value nets from {args.init_from} (aux head + optimisers fresh)")
    if path:
        ck = torch.load(path, map_location="cpu")
        if ck.get("partner_names") != names:''')
rep('''    ap.add_argument("--checkpoint", default=None)''',
'''    ap.add_argument("--no-selfplay", action="store_true",
                    help="leave the self-play seeds out of the pool (fixed, reproducible pool)")
    ap.add_argument("--init-from", default=None,
                    help="population mode: start the policy/value nets from this checkpoint")
    ap.add_argument("--checkpoint", default=None)''')
rep('''    args = ap.parse_args()
''', '''    args = ap.parse_args()
    globals()["SKIP_SELFPLAY"] = args.no_selfplay
''')
open(path, "w").write(src)
print("patched OK")
