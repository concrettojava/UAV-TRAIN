# UAV-TRAIN implementation and interruption-recovery audit

## Verified disk state

The file tree was inspected after the interrupted conversation. All major
implementation files from the previous session were present. No duplicate
copy or destructive reset was performed.

The new project is NOT a Git repository. The original source project is a
separate Git repository and had a clean working tree at final audit.
Its HEAD moved between observations (first 110e840, later 1b8ddb1),
apparently due to independent work; no write to that project was issued here.

## Completed work in this standalone package

- Self-contained env/model/training imports, no source-tree dependencies.
- Standalone runner, model snapshot format, evaluation subprocess and resume.
- GPU staging buffers and independent compiled actor inference and PPO graphs.
- Full 1024-row PPO with 10 epochs; removed gradient microbatch code path.
- Disabled Actor activation checkpointing and Python row chunking.
- Reused the sanitized observation in dual-scale map encoding.
- Fused seven-choice action decoder attention with PyTorch SDPA.
- Vectorized 16 public trajectory integration steps.
- Asynchronous fixed-seed CPU validation; TensorBoard validation mean episode
  length, success, completion, survival and return.
- Every-200-update numerical audit uses captured pre-update model plus
  already-collected action/log-prob/active samples. CPU worker owns comparison.
- Per-update train TensorBoard includes losses, KL, entropy, clipping, Actor
  learning rate, episode metrics and complete PPO-update interval in seconds.
- Full training checkpoints and restart protocol; metrics trimmed to checkpoint
  step on resume; independent evaluator guarded by a process lock.
- No requirements.txt and no automatic environment package installation.

## Checks executed

- Source syntax: 26 Python files compiled via Python built-in compile().
- Shell syntax: bash -n run.sh passed.
- CLI: python -m training.train --help returned 0; it did not start training.
- Pytest CPU tests: 6 passed, including one tiny synthetic PPO update,
  optimizer-step counts, log-prob audit, checkpoint reload, deterministic
  evaluation, fused operator numerical equivalence, log reconciliation,
  import and package audits.
- Original repository: git status --porcelain returned no modifications.
- Output directory: empty, no latest.pt and no formal GPU run created.

## Remaining explicit boundary

No compiled CUDA/PyTorch GPU graph runtime or full 8x128x10 GPU run was
executed during this audit. The local RTX 3080 was near 98 percent utilization,
so testing compilation here would interfere with other running work.
CUDA Graph capture, optimized training throughput, and eventual convergence
must be established on the deployment GPU before scientific performance
claims. CPU unit tests alone do not establish GPU numerical equivalence.

No formal training was started.
