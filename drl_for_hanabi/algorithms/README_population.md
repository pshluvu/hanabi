# Population-trained Hanabi agent (MSc thesis: Convention Discovery Under Partner Uncertainty)

Two-player Hanabi (5 colours, 5 ranks, 8 information tokens, 3 lives). The score is the
number of fireworks if any lives remain and 0 if all three are lost.
One agent (`population`) is trained with REINFORCE (or PPO) against a pool of partners and
evaluated in cross-play. Self-play is only an evaluation measure: the agent is never trained
against a copy of itself.

## Folder: `drl_for_hanabi/algorithms/`

| File | What it is |
|---|---|
| `population_training_v5.py` | Current training and evaluation script. Adds `--ppo` to v4. |
| `population_training_v4.py` | v3 plus GAE (`--gae-lambda`), belief head (`--belief-aux`), partner weights, `--batch-size`, `--selfplay-eval`. |
| `population_training_v3.py` | 20-action recurrent agent, replay buffer (`--replay-cap`, `--bc-policy`, `--bc-value`), external OBL/SAD partners. |
| `population_training_v2.py`, `population_training_recurrent.py` | Earlier versions kept for the record. |
| `patch_*.py` | Small scripts that applied each change to the training script. |
| `run_eval.sh`, `run_eval2.sh`, `run_eval3.sh`, `run_eval4.sh` | Evaluation (40 games per partner, seats alternating). |
| `run_s3.sh`, `run_s4.sh`, `run_v4.sh`, `run_v5.sh` | Training stages (see below). |
| `trend.py` | Averages the second half of a training log per partner. |
| `sanity_check_cheat_agents.py` | Diagnostic showing the PPO/VPG/SPG baselines only work with own-card access. |
| `test_*.py`, `time_partners.py` | Small tests and timing checks. |

`drl_for_hanabi/extra/decentralized/` holds the cross-process proxies for OBL and SAD
(`agents/obl`, `sad_agent_proxy.py`, `deck_sync_utils.py`) and the saved cross-play results.

## Action space and observation
20 actions: 5 discards, 5 plays, 5 colour hints, 5 rank hints. Illegal actions are masked.
Observation: the environment's vectorised representation (658 inputs). The policy is a
recurrent network built from a tanh (Elman) layer. `nn.GRUCell` crashes on this machine's
PyTorch 1.7 build.

## Partners
- Rule-based: `bram_simple`, `bram_best` (rewritten for standard partial-information rules).
- External, each run in its own process because they need PyTorch 2.4: Off-Belief Learning
  (OBL) and the Simplified Action Decoder (SAD). Pairs of two external agents cannot be run
  together, because both use player index 0.
- Removed: `simple`, `ppo`, `vpg`, `spg` (not competent under standard rules; the last three
  were trained with own-card access) and `hoad_piers` (about 0 with this agent, very slow).

## Stages
1. Single partner (`bram_simple`): works with it, fails with unseen partners.
2. Add `bram_best` (replay buffer on or off).
3. Add OBL as a training partner (SAD held out).
4. Train with SAD, hold OBL out.
5. v4 experiments (GAE, belief head, OBL oversampling) and v5 (PPO versus a REINFORCE control,
   all four partners in training).

## Running
```bash
source /home/percy/hanabi/venv/bin/activate
cd ~/hanabi/drl_for_hanabi/algorithms
nohup ./run_v5.sh v5_ppo --ppo > /dev/null 2>&1 &     # training
./run_eval4.sh v5_ppo                                   # evaluation, writes evalD_v5_ppo.log
```
Environment variables used by every script:
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 TF_ENABLE_ONEDNN_OPTS=0 KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""`.
`hoad_piers` (TensorFlow) crashes on build on this machine, so always pass
`--partners bram_simple bram_best`.

## Headline results (40 games per cell, one seed, mean score out of 25)
| Run | bram_simple | bram_best | OBL | SAD |
|---|---|---|---|---|
| Single partner (`r0_v2`) | 9.95 | 2.80 (unseen) | 2.70 (unseen) | 9.60 (unseen) |
| `s3_replay` (OBL trained, SAD held out) | 10.55 | 10.22 | 7.62 | 10.22 (unseen) |
| `s4_plain` (SAD trained, OBL held out) | 10.57 | 10.18 | 2.75 (unseen) | 13.95 |
| `v5_ppo` (all four trained) | 11.47 | 10.53 | 6.78 | 14.70 |
| `v5_ctrl` (REINFORCE, all four trained) | 11.88 | 9.65 | 6.53 | 13.65 |

Reference: `bram_simple` and `bram_best` score 16.5 and 12.9 with themselves and 11.6 with each
other. The agent's own self-play is near zero, as expected. With OBL the score stays at about
7 across every change tried (GAE, belief head, oversampling, larger batches, PPO).
Standard error per cell is about 0.5, so differences under 1 point are noise.
