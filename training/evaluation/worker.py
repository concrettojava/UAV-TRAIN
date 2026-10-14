"""Asynchronous CPU evaluator and numerical consistency auditor."""
from __future__ import annotations
import argparse
import fcntl
import math
import json
import os
from pathlib import Path
import time
import torch

from training.evaluation.evaluate import evaluate, load_policy


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def run_evaluation(snapshot, root, writer, episodes, seed_start):
    data = evaluate(snapshot, episodes=episodes, seed_start=seed_start,
                    parallel=min(16, episodes), device="cpu")
    update = int(snapshot.stem.split("-")[-1])
    summary = data["summary"]
    append_jsonl(root / "evaluations.jsonl",
                 dict(update=update, summary=summary, episodes=episodes,
                      seed_start=seed_start))
    for key in ("completion_mean", "success_rate", "survival_mean", "return_mean"):
        writer.add_scalar("validation/" + key, summary[key], update)
    writer.add_scalar("validation/mean_episode_length",
                      summary["termination_steps_mean"], update)
    score = (summary["completion_mean"], summary["success_rate"], summary["survival_mean"])
    best_file = root / "best-selection.json"
    previous = json.loads(best_file.read_text()) if best_file.exists() else None
    if previous is None or score > tuple(previous["score"]):
        best = dict(update=update, score=score, summary=summary)
        temp = root / "best.pt.tmp"
        import shutil
        shutil.copyfile(snapshot, temp)
        temp.replace(root / "best.pt")
        best_file.write_text(json.dumps(best, indent=2), encoding="utf-8")
    writer.flush()


def run_audit(path, root):
    item = torch.load(path, map_location="cpu", weights_only=True)
    model, _, _ = load_policy(item["checkpoint"], device="cpu")
    obs = item["obs"].float()
    actions = item["actions"].long()
    active = item["active"].bool()
    expected = item["logp"].float()
    with torch.inference_mode():
        reference = model.distribution(obs).log_prob(actions).float()
    delta = (reference - expected).abs()
    observed = delta[active]
    finite = all(bool(torch.isfinite(x).all()) for x in model.parameters())
    finite = (finite and bool(torch.isfinite(reference).all())
              and bool(torch.isfinite(expected).all()))
    mx = float(observed.max()) if observed.numel() else 0.
    mean = float(observed.mean()) if observed.numel() else 0.
    append_jsonl(root / "numerical-audits.jsonl",
                 dict(update=item["update"],
                      max_logp_difference=mx if math.isfinite(mx) else None,
                      mean_logp_difference=mean if math.isfinite(mean) else None,
                      active_samples=int(observed.numel()),
                      finite=finite, reference_backend="cpu_eager",
                      observed_backend="rollout_collected",
                      note="Audit uses matching pre-update model and collected log-probs; CPU/GPU numeric drift is diagnostic-only"))
    path.unlink()
    Path(item["checkpoint"]).unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--parent-pid", type=int, required=True)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seed-start", type=int, default=1401000)
    args = p.parse_args()
    root = args.run.resolve()
    # A previous evaluator may still be draining snapshots after a restart.
    lock = (root / "evaluation-worker.lock").open("a+b")
    fcntl.flock(lock, fcntl.LOCK_EX)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(root.parent.parent / "tensorboard" / "MBDT-DualScale-256-FULL"))
    try:
        while True:
            pending = root / "pending"
            activity = False
            for snapshot in sorted(pending.glob("validation-*.pt")):
                try:
                    run_evaluation(snapshot, root, writer, args.episodes, args.seed_start)
                    snapshot.unlink()
                except Exception as error:
                    append_jsonl(root / "worker-errors.jsonl",
                                 dict(file=snapshot.name, error=repr(error)))
                    snapshot.rename(snapshot.with_suffix(".failed"))
                activity = True
            for path in sorted(pending.glob("audit-*.pt")):
                try:
                    run_audit(path, root)
                except Exception as error:
                    append_jsonl(root / "worker-errors.jsonl",
                                 dict(file=path.name, error=repr(error)))
                    path.rename(path.with_suffix(".failed"))
                activity = True
            if not activity:
                try:
                    os.kill(args.parent_pid, 0)
                except ProcessLookupError:
                    if not list(pending.glob("*.pt")):
                        break
                time.sleep(2)
    finally:
        writer.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
