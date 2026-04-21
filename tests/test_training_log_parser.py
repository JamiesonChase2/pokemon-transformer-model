from scripts.plot_training_log import parse_training_log_lines


def test_parse_training_log_lines_deduplicates_repeated_notebook_output():
    parsed = parse_training_log_lines(
        [
            "[iter 3] start selfplay_battles=10 selfplay_steps=32768 train_steps=0 train_epochs=3 eval_battles=50 total_env_steps=12345 lr=0.0001 reward_terminal=(+1.000,-1.000) reward_faint=(-0.100,+0.100) max_turns_before_forfeit=300",
            "[iter 3] train start steps=27 requested_steps=auto batch=512 replay=33138 target_kl=0.020 epochs=3 grad_accum=8 effective_batch=4096",
            "[iter 3] train_step=10/27 loss=1.1127 policy=-0.0188 value=2.3114 entropy=1.2112 kl=0.0253 clip_frac=0.251",
            "[iter 3] train_step=10/27 loss=1.1127 policy=-0.0188 value=2.3114 entropy=1.2112 kl=0.0253 clip_frac=0.251",
            "[iter 3] self-play done battles=100 steps=32768 episodes=100 replay=33138 elapsed=120.0s",
            "[iter 3] eval done elapsed=45.0s win_rate=0.640",
            "iter=3 replay=33138 selfplay_wlt=578/262/0 loss=1.1194 policy_loss=0.0031 value_loss=2.2792 entropy=1.1629 kl=0.0135 clip_frac=0.144 eval_win_rate=0.640 eval_ci=[0.501,0.759] promoted=True iter_elapsed=746.5s",
            "iter=3 replay=33138 selfplay_wlt=578/262/0 loss=1.1194 policy_loss=0.0031 value_loss=2.2792 entropy=1.1629 kl=0.0135 clip_frac=0.144 eval_win_rate=0.640 eval_ci=[0.501,0.759] promoted=True iter_elapsed=746.5s",
        ]
    )

    assert len(parsed.iteration_starts) == 1
    assert len(parsed.train_starts) == 1
    assert len(parsed.train_steps) == 1
    assert len(parsed.selfplay_done) == 1
    assert len(parsed.eval_done) == 1
    assert len(parsed.summaries) == 1

    summary = parsed.summaries[0]
    assert summary.iteration == 3
    assert summary.eval_win_rate == 0.640
    assert summary.promoted is True


def test_parse_training_log_lines_handles_reward_faint_off():
    parsed = parse_training_log_lines(
        [
            "[iter 1] start selfplay_battles=8 selfplay_steps=4096 train_steps=0 train_epochs=2 eval_battles=16 total_env_steps=0 lr=1e-07 reward_terminal=(+1.000,-1.000) reward_faint=off max_turns_before_forfeit=500",
        ]
    )

    assert len(parsed.iteration_starts) == 1
    assert parsed.iteration_starts[0].reward_faint == "off"
