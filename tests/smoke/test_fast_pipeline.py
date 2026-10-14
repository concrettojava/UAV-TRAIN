"""Isolated CPU smoke tests. Never starts formal training or compiles CUDA kernels."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from envs import EnvConfig
from training.algo.buffer import ACTIVE, LOGP
from training.algo.mappo import MAPPOConfig, MAPPOTrainer
from training.evaluation.worker import run_audit, run_evaluation
from training.train import policy_snapshot
from torch.utils.tensorboard import SummaryWriter


def test_full_batch_ppo_and_cpu_worker(tmp_path):
    torch.set_num_threads(1)
    cfg = MAPPOConfig(
        num_envs=2, rollout_steps=2, hidden=64, epochs=1, minibatch_size=4,
        seed=33, torch_threads=1, env_workers=1,
        model_name="mbdt_dualscale", collection_mode="fragment",
        value_normalization=True, belief_grid=32)
    env_cfg = EnvConfig(scenario="contested", reward_profile="task_v2",
                        state_layout="full_v1", max_steps=2, time_limit_mode="terminal")
    trainer = MAPPOTrainer(cfg, env_cfg, device="cpu")
    actor_steps = [0]
    critic_steps = [0]
    actual_actor_step = trainer.actor_opt.step
    actual_critic_step = trainer.critic_opt.step

    def count_actor(*args, **kwargs):
        actor_steps[0] += 1
        return actual_actor_step(*args, **kwargs)

    def count_critic(*args, **kwargs):
        critic_steps[0] += 1
        return actual_critic_step(*args, **kwargs)

    trainer.actor_opt.step = count_actor
    trainer.critic_opt.step = count_critic
    audit_file = tmp_path / "audit-00000001.pt"
    reference_file = tmp_path / "reference-00000001.pt"

    def before_update(current):
        assert current.buffer.valid_steps == 2
        assert current.updates == 0
        torch.save(policy_snapshot(current), reference_file)
        torch.save({
            "update": 1, "checkpoint": str(reference_file),
            "obs": torch.from_numpy(current.buffer.obs[0].copy()),
            "actions": torch.from_numpy(current.buffer.actions[0].copy()),
            "logp": torch.from_numpy(current.buffer.data[0, ..., LOGP].copy()),
            "active": torch.from_numpy(current.buffer.data[0, ..., ACTIVE].copy()),
        }, audit_file)

    try:
        metrics = trainer.train_iteration(pre_update_hook=before_update)
        assert trainer.updates == 1
        assert trainer.env_steps == 4
        assert actor_steps == [1] and critic_steps == [1]
        for name in ("actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
            assert np.isfinite(metrics[name]), name
        assert len(trainer.last_completed_episodes) > 0

        run_audit(audit_file, tmp_path)
        assert not audit_file.exists()
        assert not reference_file.exists()
        row = json.loads((tmp_path / "numerical-audits.jsonl").read_text().splitlines()[0])
        assert row["update"] == 1
        assert row["finite"] is True
        assert row["active_samples"] > 0
        assert row["max_logp_difference"] < 0.001

        checkpoint = tmp_path / "latest.pt"
        trainer.save(checkpoint)
        restored = MAPPOTrainer.load(checkpoint, device="cpu")
        try:
            assert restored.updates == trainer.updates
            assert restored.env_steps == trainer.env_steps
            assert restored.cfg.epochs == cfg.epochs
        finally:
            restored.close()

        snapshot = tmp_path / "validation-00000001.pt"
        torch.save(policy_snapshot(trainer), snapshot)
        with SummaryWriter(str(tmp_path / "tb")) as writer:
            run_evaluation(snapshot, tmp_path, writer, episodes=2, seed_start=100000)
        assert (tmp_path / "evaluations.jsonl").exists()
        assert (tmp_path / "best.pt").exists()
        summary = json.loads((tmp_path / "evaluations.jsonl").read_text().splitlines()[0])["summary"]
        assert "termination_steps_mean" in summary
        assert 1 <= summary["termination_steps_mean"] <= 2
    finally:
        trainer.close()


def test_project_imports_and_no_microbatch():
    root = Path(__file__).resolve().parents[2]
    import inspect
    from models.mbdt_perf import PerformanceActor, DualScaleActor
    from training.fast_compile import configure_fast_execution

    text = inspect.getsource(PerformanceActor.forward)
    assert "return self._forward(x)" in text
    assert "checkpoint(" not in text
    assert "pre_update_hook" in inspect.signature(MAPPOTrainer.train_iteration).parameters
    assert (root / "training/configs/fast.json").is_file()
    cfg = json.loads((root / "training/configs/fast.json").read_text())
    assert (cfg["num_envs"], cfg["rollout_steps"], cfg["ppo_epochs"], cfg["microbatch_rows"]) == (8,128,10,0)


def test_resume_metrics_reconciliation(tmp_path):
    from training.train import reconcile_training_log
    f = tmp_path / "metrics.jsonl"
    records = [dict(update=i,actor_loss=float(i)) for i in (1,2,3)]
    f.write_text("".join(json.dumps(x) + "\n" for x in records))
    assert reconcile_training_log(f,2) == 1
    left = [json.loads(line) for line in f.read_text().splitlines()]
    assert [row["update"] for row in left] == [1,2]
    assert reconcile_training_log(f,2) == 0
