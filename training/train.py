"""From-scratch MBDT-DualScale-256 high-throughput MAPPO training."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from envs import EnvConfig
from training.algo.mappo import MAPPOConfig, MAPPOTrainer
from training.algo.buffer import LOGP, ACTIVE
from training.fast_compile import configure_fast_execution

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/runs/MBDT-DualScale-256"
TB = ROOT / "outputs/tensorboard/MBDT-DualScale-256-FULL"
STOP = False


def signal_stop(sig, _):
    global STOP
    STOP = True


def atomic_save(value, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def policy_snapshot(trainer):
    return dict(schema="uavmarl.mappo.v1",
                config=asdict(trainer.cfg), env_config=asdict(trainer.env_cfg),
                model={key: val.detach().cpu() for key, val in trainer.model.state_dict().items()})


def reconcile_training_log(path, checkpoint_update):
    # A crash can append a JSONL row before the next durable checkpoint.
    # Keep only the authoritative prefix; TensorBoard uses purge_step on resume.
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    keep = []
    discarded = 0
    last = -1
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        step = int(record["update"])
        if step <= last:
            raise ValueError("non-monotonic metrics history")
        last = step
        if step <= checkpoint_update:
            keep.append(line)
        else:
            discarded += 1
    if discarded:
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        tmp.replace(path)
    return discarded


def log_scalar(writer, tag, value, step):
    if value is not None and np.isfinite(value):
        writer.add_scalar(tag, float(value), step)


def training_scores(episodes):
    if not episodes:
        return {}
    return dict(
        return_mean=float(np.mean([e["return"] for e in episodes])),
        completion_mean=float(np.mean([e["completion"] for e in episodes])),
        success_rate=float(np.mean([e["completion"] == 1. for e in episodes])),
        survival_mean=float(np.mean([e["survival"] for e in episodes])),
        mean_episode_length=float(np.mean([e["steps"] for e in episodes])))


def run(args):
    RUN.mkdir(parents=True, exist_ok=True)
    TB.mkdir(parents=True, exist_ok=True)
    pending = RUN / "pending"
    pending.mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    if args.resume:
        if not (RUN / "latest.pt").exists():
            raise FileNotFoundError("cannot resume: missing latest.pt")
        trainer = MAPPOTrainer.load(RUN / "latest.pt", device=args.device)
        saved_manifest = json.loads((RUN / "manifest.json").read_text())
        if (saved_manifest.get("compile") != (not args.no_compile)
                or saved_manifest.get("precision") != args.precision
                or saved_manifest.get("eval_every") != args.eval_every
                or saved_manifest.get("eval_episodes") != args.eval_episodes
                or saved_manifest.get("eval_seed") != args.eval_seed
                or saved_manifest.get("save_every") != args.save_every
                or saved_manifest.get("config", {}).get("seed") != trainer.cfg.seed):
            raise ValueError("resume flags disagree with original training protocol")
        discarded = reconcile_training_log(RUN / "metrics.jsonl",trainer.updates)
        if discarded:
            print(json.dumps(dict(event="discarded_uncheckpointed_metric_rows",
                                  count=discarded, checkpoint_update=trainer.updates)),
                  flush=True)
    else:
        if (RUN / "latest.pt").exists() or (RUN / "metrics.jsonl").exists():
            raise FileExistsError("run already exists; use --resume or an empty run folder")
        cfg = MAPPOConfig(num_envs=8, rollout_steps=128, hidden=256, epochs=10,
                          minibatch_size=1024, lr=4e-5, gamma=.95, gae_lambda=.95,
                          clip=.1, value_clip=.1, entropy_coef=.01, max_grad_norm=.5,
                          seed=args.seed, torch_threads=1, value_normalization=True,
                          env_workers=1, model_name="mbdt_dualscale",
                          collection_mode="fragment", belief_grid=32)
        env_cfg = EnvConfig(scenario="contested", reward_profile="task_v2",
                            state_layout="full_v1", max_steps=200,
                            time_limit_mode="terminal")
        trainer = MAPPOTrainer(cfg, env_cfg, device=args.device)
        for group in trainer.actor_opt.param_groups:
            group["lr"] = args.actor_lr

    if trainer.cfg.epochs != 10 or trainer.cfg.num_envs != 8 or trainer.cfg.rollout_steps != 128:
        raise ValueError("this fast run is fixed to 8 env, 128 steps, 10 PPO epochs")
    if trainer.cfg.minibatch_size != trainer.cfg.num_envs * trainer.cfg.rollout_steps:
        raise ValueError("PPO must use one complete 1024-row batch")

    configure_fast_execution(trainer, enabled=not args.no_compile, precision=args.precision)
    manifest = dict(name="MBDT-DualScale-256", from_scratch=not args.resume,
                    config=asdict(trainer.cfg), env_config=asdict(trainer.env_cfg),
                    compile=not args.no_compile, precision=args.precision,
                    actor_lr=trainer.actor_opt.param_groups[0]["lr"],
                    eval_every=args.eval_every, eval_episodes=args.eval_episodes,
                    eval_seed=args.eval_seed, save_every=args.save_every,
                    audit_every=200,
                    model_sha256=hashlib.sha256((ROOT / "models/mbdt_perf.py").read_bytes()).hexdigest())
    if not args.resume:
        (RUN / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(TB), purge_step=trainer.updates + 1 if args.resume else None,
                           max_queue=1000, flush_secs=30)
    worker_log = (RUN / "worker.log").open("a", encoding="utf-8")
    worker = subprocess.Popen([
        sys.executable, "-m", "training.evaluation.worker",
        "--run", str(RUN), "--parent-pid", str(os.getpid()),
        "--episodes", str(args.eval_episodes), "--seed-start", str(args.eval_seed)],
        cwd=ROOT, stdout=worker_log, stderr=subprocess.STDOUT)
    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)

    try:
        while not STOP and not (RUN / "stop-request").exists():
            if args.updates and trainer.updates >= args.updates:
                break
            started = time.perf_counter()
            audit_due = (trainer.updates + 1) % 200 == 0

            def capture_audit(current):
                # Collected actions and log-probs correspond to the parameters
                # BEFORE this PPO step. Reuse them; do not run an extra actor pass.
                update = current.updates + 1
                reference = pending / f"reference-{update:08d}.pt"
                snapshot = policy_snapshot(current)
                atomic_save(snapshot, reference)
                sample = dict(
                    update=update, checkpoint=str(reference),
                    obs=torch.from_numpy(current.buffer.obs[0].copy()),
                    actions=torch.from_numpy(current.buffer.actions[0].copy()),
                    logp=torch.from_numpy(current.buffer.data[0, ..., LOGP].copy()),
                    active=torch.from_numpy(current.buffer.data[0, ..., ACTIVE].copy()))
                atomic_save(sample, pending / f"audit-{update:08d}.pt")

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(args.precision == "bf16" and trainer.device.type == "cuda")):
                stats = trainer.train_iteration(
                    pre_update_hook=capture_audit if audit_due else None)
            elapsed = time.perf_counter() - started
            update = trainer.updates
            episode_scores = training_scores(trainer.last_completed_episodes)
            core = dict(update=update, env_steps=trainer.env_steps,
                        actor_loss=stats["actor_loss"], critic_loss=stats["value_loss"],
                        approx_kl=stats["approx_kl"], entropy=stats["entropy"],
                        clip_fraction=stats["clip_fraction"],
                        active_fraction=stats["active_fraction"],
                        actor_lr=trainer.actor_opt.param_groups[0]["lr"],
                        update_interval_s=elapsed,
                        completed_episodes=len(trainer.last_completed_episodes),
                        **episode_scores)
            with (RUN / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(core, ensure_ascii=False, allow_nan=False) + "\n")
            for key in ("actor_loss", "critic_loss", "approx_kl", "entropy",
                        "clip_fraction", "active_fraction", "actor_lr",
                        "update_interval_s", "return_mean", "completion_mean",
                        "success_rate", "survival_mean", "mean_episode_length"):
                log_scalar(writer, "train/" + key, core.get(key), update)

            if update == 1 or update % args.save_every == 0:
                trainer.save(RUN / "latest.pt")
            if update % args.eval_every == 0:
                atomic_save(policy_snapshot(trainer),
                            pending / f"validation-{update:08d}.pt")

            if update <= 3 or update % 10 == 0:
                print(json.dumps(dict(update=update, update_interval_s=round(elapsed, 3),
                                      env_steps=trainer.env_steps)), flush=True)
    finally:
        trainer.save(RUN / "latest.pt")
        writer.flush()
        writer.close()
        trainer.close()
        worker_log.close()
        print(f"Training ended at update {trainer.updates}; evaluator PID {worker.pid}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--updates", type=int, default=0, help="0 means run until stopped")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--precision", choices=("fp32", "tf32", "bf16"), default="tf32")
    p.add_argument("--actor-lr", type=float, default=2e-5)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=64)
    p.add_argument("--eval-seed", type=int, default=1401000)
    args = p.parse_args()
    if min(args.save_every, args.eval_every, args.eval_episodes) < 1:
        p.error("intervals must be positive")
    if args.eval_every % args.save_every != 0:
        p.error("evaluation snapshots require a durable checkpoint at the same update")
    run(args)


if __name__ == "__main__":
    main()
