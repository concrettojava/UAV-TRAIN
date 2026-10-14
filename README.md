# UAV-TRAIN: MBDT-DualScale-256 speed-first training

Portable, standalone research training package. The original source tree at
`/home/uav-autodl` is not modified by this project.

## Training entry point

Use an existing Python environment with PyTorch, NumPy and TensorBoard.
No requirements.txt or automatic package installation is included.

```sh
cd /home/UAV-TRAIN
./run.sh
```

Running the command above starts a new formal run. **Do not run until deployment is approved.**
Use `./run.sh --resume` to load `outputs/runs/MBDT-DualScale-256/latest.pt`.
Use `./run.sh --updates N` only for explicitly authorized bounded runs.
A `stop-request` file in the run directory signals a graceful shutdown.

## Configuration

- Eight batched NumPy environments, 128 simulation steps per rollout.
- 10 PPO epochs, one full 1024-row batch, no gradient microbatches.
- Dual-scale evidence and seven physical trajectory action choices.
- Fast-path Actor inference compiled independently of Actor gradient computation;
  full Actor / Critic PPO and per-agent optimizer steps compiled.
- No activation checkpointing or Python Actor chunking by default.
- FP32 parameters, TF32 by default; optional BF16 autocast. PPO probability
  calculations use FP32 logits.
- Core training metrics: Actor/Critic losses, KL, entropy, clipping,
  active fraction, Actor LR, return, completion/success/survival and
  seconds per complete PPO update.
- Independent CPU validation process, every 50 updates by default; reports
  mean episode length in TensorBoard.
- CPU numerical audit every 200 updates uses the pre-update model with
  already-collected action/log-prob samples. Numeric drift is logged, not fatal.
- Full trainer state, optimizer state, environment RNG and CUDA RNG stored in
  `latest.pt` for resume.

## Layout

```text
models/                      Network and computational operators
envs/                        Batched simulation and configuration
training/algo/               MAPPO, rollout storage, independent AgentAdam
training/evaluation/         CPU evaluation and asynchronous numerical audit
training/configs/            Recorded configuration
training/fast_compile.py     Compiler and device performance configuration
training/train.py            Formal training entry point (not run yet)
tests/                       Isolated smoke and numerical checks
outputs/runs/MBDT-DualScale-256/
outputs/tensorboard/MBDT-DualScale-256-FULL/
run.sh
```

## Boundaries and remaining validation

Only this package is editable. The source project and its current long run
remain untouched. A server-side GPU test is still required to establish
TorchInductor/CUDA Graph correctness and actual throughput. No speedup is
claimed until that test is completed.
