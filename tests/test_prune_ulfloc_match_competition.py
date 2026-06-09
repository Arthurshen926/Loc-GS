from loc_gs.scripts.prune_ulfloc_match_competition import build_argparser


def test_prune_ulfloc_match_competition_argparser_accepts_required_args():
    args = build_argparser().parse_args(
        [
            "--input_model_path",
            "/models/in",
            "--input_log_dir",
            "/models/in/log",
            "--output_model_path",
            "/models/out",
            "--cfg",
            "/models/in/log/config.yaml",
            "--output_log_name",
            "match_competition_pruned",
            "--risk_scores",
            "/scores/stealer_scores.pkl",
            "--sparse_validation_profile",
            "/profiles/sparse_pnp_validation_profile.json",
            "--min_risk_score",
            "0.1",
            "--max_prune_count",
            "64",
        ]
    )

    assert str(args.input_model_path) == "/models/in"
    assert str(args.input_log_dir) == "/models/in/log"
    assert str(args.output_model_path) == "/models/out"
    assert str(args.risk_scores) == "/scores/stealer_scores.pkl"
    assert str(args.sparse_validation_profile) == "/profiles/sparse_pnp_validation_profile.json"
    assert args.min_risk_score == 0.1
    assert args.max_prune_count == 64
