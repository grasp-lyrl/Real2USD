"""End-to-end harness smoke test on the synthetic backend (no data/GPU needed)."""

import json

from r2s3d_core.eval.run import build_parser, run


def test_oracle_perfect(tmp_path):
    args = build_parser().parse_args([
        "--source", "synthetic", "--scene", "synthetic", "--method", "oracle",
        "--out", str(tmp_path / "oracle"),
    ])
    out = run(args)
    rec = json.loads(out.read_text())
    agg = rec["metrics"]["aggregate"]
    assert agg["f1"] == 1.0
    assert agg["scan2cad_accuracy"] == 1.0
    assert agg["centroid_err_median_m"] < 1e-9
    assert agg["rotation_err_median_deg"] < 1e-6
    assert agg["duplicate_rate"] == 0.0
    assert rec["git_sha"] and rec["created_at"] and "config" in rec


def test_oracle_noisy_degrades(tmp_path):
    args = build_parser().parse_args([
        "--source", "synthetic", "--scene", "synthetic", "--method", "oracle_noisy",
        "--trans-noise-m", "0.1", "--out", str(tmp_path / "noisy"),
    ])
    out = run(args)
    agg = json.loads(out.read_text())["metrics"]["aggregate"]
    assert agg["centroid_err_median_m"] > 0.0
    assert agg["f1"] == 1.0  # still all matched at IoU 0.25
