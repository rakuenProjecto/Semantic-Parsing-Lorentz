import json
import subprocess
import sys
from pathlib import Path

from scripts.auto_batch_probe import probe_batches
from scripts.diagnose_experiment import diagnose
from scripts.summarize_latest_experiment import summarize, write_reports


def test_summarize_latest_experiment_reads_sample_diagnostics(tmp_path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "diagnostics_final.json").write_text(
        json.dumps(
            {
                "pred_abs_c_mean": 0.8,
                "pred_abs_c_std": 0.1,
                "curvature_min_fraction": 0.0,
                "curvature_mean_to_min_ratio": 16.0,
                "jac_frob_mean": 2.0,
                "jac_frob_std": 0.2,
            }
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "epoch=1 train_loss=1.0 train_acc=0.5 val_loss=0.9 val_acc=0.7\n"
        "training complete best_val_loss=0.9 checkpoint=x\n",
        encoding="utf-8",
    )

    summary = summarize(output_dir, log_path, cycle_id="1")
    write_reports(summary, tmp_path / "reports", cycle_id="1", timestamp="t")

    assert summary["pred_abs_c_mean"] == 0.8
    assert summary["final_val_acc"] == 0.7
    assert (tmp_path / "reports" / "latest_experiment_summary.json").exists()


def test_diagnose_experiment_detects_collapse() -> None:
    diagnosis = diagnose(
        {
            "curvature_min_fraction": 1.0,
            "pred_abs_c_mean": 0.051,
            "pred_abs_c_std": 0.0005,
            "curvature_mean_to_min_ratio": 1.03,
            "jac_frob_mean": 2.0,
            "final_val_loss": 0.5,
            "final_val_acc": 1.0,
        }
    )

    issue_types = {issue["type"] for issue in diagnosis["issues"]}
    assert diagnosis["status"] == "needs_fix"
    assert "curvature_collapse" in issue_types
    assert "floor_sticking" in issue_types


def test_auto_batch_probe_dry_run() -> None:
    payload = probe_batches([2, 4], "configs/bert_true_jacobian_anti_collapse_debug.yaml", dry_run=True)

    assert payload["safe_batch_size"] == 4
    assert all(result["status"] == "dry_run" for result in payload["results"])


def test_autonomous_experiment_loop_dry_run_creates_plan() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/autonomous_experiment_loop.py", "--mode", "debug", "--dry-run"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    assert "dry_run_planned" in result.stdout


def test_run_experiment_cycle_shell_syntax() -> None:
    subprocess.run(["bash", "-n", "scripts/run_experiment_cycle.sh"], check=True)
