import re, sys
from collections import defaultdict
for f in ("v5_ppo.log", "v5_ctrl.log"):
    acc = defaultdict(list)
    for line in open(f):
        m = re.match(r"epoch\s+(\d+)", line)
        if not m or int(m.group(1)) < 500:
            continue
        for name, val in re.findall(r"(bram_best|bram_simple|obl|sad):([\d.]+)", line):
            acc[name].append(float(val))
    print(f, {k: round(sum(v) / len(v), 2) for k, v in acc.items()}, "lines:", len(acc["obl"]))
