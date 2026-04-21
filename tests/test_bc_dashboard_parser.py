import tempfile
from pathlib import Path

from scripts.dashboard_imitation_runs import _collect_segments, parse_bc_log_lines, summarize_run


def test_parse_bc_log_lines_deduplicates_repeated_output_and_handles_nan_eval():
    parsed = parse_bc_log_lines(
        [
            "[upd 25] start demo_battles=80 steps_per_update=32768 train_steps=0 train_epochs=3 max_concurrent_battles=4 servers=8 workers=8 total_env_steps=819200 lr=0.0001 best_eval_win_rate=0.550 label_smoothing=0.100",
            "[upd 25] train start steps=24 requested_steps=auto batch=4096 replay=32768 epochs=3 grad_accum=1 effective_batch=4096",
            "[upd 25] train_step=10/24 loss=0.8821 policy=0.8821 value=0.0000 entropy=0.4112 acc=0.713",
            "[upd 25] train_step=10/24 loss=0.8821 policy=0.8821 value=0.0000 entropy=0.4112 acc=0.713",
            "[upd 25] demo done battles=80 steps=33210 replay=33210 skipped=2 elapsed=180.5s",
            "[upd 25] train done elapsed=74.2s",
            "upd=25 replay=33210 demo_steps=33210 skipped=2 loss=0.8012 policy_loss=0.8012 value_loss=0.0000 entropy=0.3920 train_accuracy=0.742 eval_win_rate=nan best_eval_win_rate=0.550 promoted=False upd_elapsed=256.0s",
            "upd=25 replay=33210 demo_steps=33210 skipped=2 loss=0.8012 policy_loss=0.8012 value_loss=0.0000 entropy=0.3920 train_accuracy=0.742 eval_win_rate=nan best_eval_win_rate=0.550 promoted=False upd_elapsed=256.0s",
            "[upd 50] start demo_battles=80 steps_per_update=32768 train_steps=0 train_epochs=3 max_concurrent_battles=4 servers=8 workers=8 total_env_steps=1638400 lr=0.0001 best_eval_win_rate=0.610 label_smoothing=0.100",
            "[upd 50] eval done elapsed=45.0s win_rate=0.640 score_rate=0.640 ci95=[0.501,0.759]",
            "upd=50 replay=33001 demo_steps=33001 skipped=0 loss=0.5520 policy_loss=0.5520 value_loss=0.0000 entropy=0.2510 train_accuracy=0.881 eval_win_rate=0.640 best_eval_win_rate=0.640 promoted=True upd_elapsed=241.5s",
        ]
    )

    assert len(parsed.update_starts) == 2
    assert len(parsed.train_steps) == 1
    assert len(parsed.demo_done) == 1
    assert len(parsed.eval_done) == 1
    assert len(parsed.summaries) == 2
    assert parsed.summaries[0].update == 25
    assert parsed.summaries[1].eval_win_rate == 0.640
    assert parsed.summaries[1].promoted is True

    summary = summarize_run(parsed)
    assert summary.updates == 2
    assert summary.best_eval_win_rate == 0.640
    assert summary.promotions == 1


def test_collect_segments_tracks_resumed_log_boundaries():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        run1 = root / "run1.log"
        run2 = root / "run2.log"
        run1.write_text(
            "\n".join(
                [
                    "[upd 1] start demo_battles=80 steps_per_update=32768 train_steps=0 train_epochs=3 max_concurrent_battles=4 servers=8 workers=8 total_env_steps=0 lr=1e-4 best_eval_win_rate=-1.000 label_smoothing=0.100",
                    "upd=1 replay=33000 demo_steps=33000 skipped=0 loss=1.2000 policy_loss=1.2000 value_loss=0.0000 entropy=1.1000 train_accuracy=0.400 eval_win_rate=nan best_eval_win_rate=-1.000 promoted=False upd_elapsed=260.0s",
                    "[upd 2] start demo_battles=80 steps_per_update=32768 train_steps=0 train_epochs=3 max_concurrent_battles=4 servers=8 workers=8 total_env_steps=33000 lr=1e-4 best_eval_win_rate=-1.000 label_smoothing=0.100",
                    "upd=2 replay=33100 demo_steps=33100 skipped=0 loss=1.1000 policy_loss=1.1000 value_loss=0.0000 entropy=1.0000 train_accuracy=0.450 eval_win_rate=nan best_eval_win_rate=-1.000 promoted=False upd_elapsed=255.0s",
                ]
            ),
            encoding="utf-8",
        )
        run2.write_text(
            "\n".join(
                [
                    "[upd 3] start demo_battles=80 steps_per_update=32768 train_steps=0 train_epochs=3 max_concurrent_battles=4 servers=8 workers=8 total_env_steps=66100 lr=1e-4 best_eval_win_rate=-1.000 label_smoothing=0.100",
                    "upd=3 replay=33200 demo_steps=33200 skipped=1 loss=1.0000 policy_loss=1.0000 value_loss=0.0000 entropy=0.9000 train_accuracy=0.500 eval_win_rate=nan best_eval_win_rate=-1.000 promoted=False upd_elapsed=250.0s",
                    "[upd 4] start demo_battles=80 steps_per_update=32768 train_steps=0 train_epochs=3 max_concurrent_battles=4 servers=8 workers=8 total_env_steps=99300 lr=1e-4 best_eval_win_rate=-1.000 label_smoothing=0.100",
                    "upd=4 replay=33300 demo_steps=33300 skipped=1 loss=0.9000 policy_loss=0.9000 value_loss=0.0000 entropy=0.8000 train_accuracy=0.550 eval_win_rate=nan best_eval_win_rate=-1.000 promoted=False upd_elapsed=245.0s",
                ]
            ),
            encoding="utf-8",
        )

        segments = _collect_segments([run1, run2], ["run1", "run2"])

    assert len(segments) == 2
    assert segments[0].label == "run1"
    assert segments[0].first_update == 1
    assert segments[0].last_update == 2
    assert segments[1].label == "run2"
    assert segments[1].first_update == 3
    assert segments[1].last_update == 4
