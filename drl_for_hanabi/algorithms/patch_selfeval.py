import sys, shutil
path = sys.argv[1]
src = open(path).read()
if "selfplay_eval" in src:
    print("already patched")
    sys.exit(0)
shutil.copy(path, path + ".bak_selfeval")

a1 = '    ap.add_argument("--checkpoint", default=None)\n'
b1 = a1 + '    ap.add_argument("--selfplay-eval", action="store_true")\n'
assert src.count(a1) == 1, "anchor 1 not found"
src = src.replace(a1, b1)

a2 = '    ext_pool = {}\n'
b2 = a2 + (
    '    self_pool = {}\n'
    '    if args.selfplay_eval:\n'
    '        p = args.checkpoint or os.path.join(run_dir, "policy_final.pth")\n'
    '        self_pool["self(copy)"] = FrozenPolicyPartner(p, explore=False)\n'
    '        print("pool: self-play copy loaded (evaluation only)")\n')
assert src.count(a2) == 1, "anchor 2 not found"
src = src.replace(a2, b2)

a3 = '("EXTERNAL", ext_pool)):'
b3 = '("EXTERNAL", ext_pool),\n                        ("SELF-PLAY", self_pool)):'
assert src.count(a3) == 1, "anchor 3 not found"
src = src.replace(a3, b3)

open(path, "w").write(src)
print("patched OK; backup at", path + ".bak_selfeval")
