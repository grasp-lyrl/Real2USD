import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _load_rows(match_csv: Path):
    with open(match_csv, newline="") as f:
        return list(csv.DictReader(f))


def _latest_match_csv(results_root: Path):
    by_run = results_root / "by_run"
    files = sorted(by_run.glob("*_match_details.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def suggest(rows, min_iou: float, min_support: int):
    alias_votes = defaultdict(Counter)  # raw -> canonical -> count
    alias_ious = defaultdict(list)  # (raw, canonical) -> [iou...]
    unresolved = Counter()
    unresolved_with_gt = Counter()

    for r in rows:
        if r.get("status") != "unmatched_pred":
            continue
        raw = (r.get("pred_label_raw") or "").strip().lower()
        pred_can = (r.get("pred_label_canonical") or "").strip().lower()
        gt_can = (r.get("gt_label_canonical") or "").strip().lower()
        iou = _to_float(r.get("iou"), 0.0)

        if not raw:
            continue

        if not pred_can:
            unresolved[raw] += 1
            if gt_can:
                unresolved_with_gt[(raw, gt_can)] += 1

        # Alias candidate: unresolved raw label, non-empty GT canonical, enough overlap signal
        if not pred_can and gt_can and iou >= min_iou:
            alias_votes[raw][gt_can] += 1
            alias_ious[(raw, gt_can)].append(iou)

    alias_suggestions = []
    for raw, votes in alias_votes.items():
        top_can, support = votes.most_common(1)[0]
        if support < min_support:
            continue
        ious = alias_ious.get((raw, top_can), [])
        alias_suggestions.append(
            {
                "raw_label": raw,
                "suggested_canonical": top_can,
                "support": int(support),
                "mean_iou": (sum(ious) / len(ious)) if ious else 0.0,
                "vote_breakdown": dict(votes),
            }
        )
    alias_suggestions.sort(key=lambda x: (-x["support"], -x["mean_iou"], x["raw_label"]))

    ignore_candidates = [
        {"raw_label": k, "count": int(v)}
        for k, v in unresolved.most_common()
        if v >= min_support and k not in {x["raw_label"] for x in alias_suggestions}
    ]

    unresolved_gt_hints = [
        {"raw_label": k[0], "nearest_gt_canonical": k[1], "count": int(v)}
        for k, v in unresolved_with_gt.most_common()
    ]

    return {
        "alias_suggestions": alias_suggestions,
        "ignore_candidates": ignore_candidates,
        "unresolved_with_gt_hints": unresolved_gt_hints,
    }


def main():
    parser = argparse.ArgumentParser(description="Suggest label aliases from eval match_details CSV.")
    parser.add_argument("--match-csv", default=None, help="Path to *_match_details.csv")
    parser.add_argument(
        "--results-root",
        default=str((Path(__file__).resolve().parent / "results").resolve()),
        help="Used when --match-csv is omitted; selects latest *_match_details.csv in by_run/",
    )
    parser.add_argument("--min-iou", type=float, default=0.03, help="Minimum IoU for alias vote")
    parser.add_argument("--min-support", type=int, default=2, help="Minimum votes/count for suggestions")
    parser.add_argument("--output-json", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    match_csv = Path(args.match_csv) if args.match_csv else _latest_match_csv(Path(args.results_root))
    if match_csv is None or not match_csv.exists():
        raise SystemExit("[ERR] No match_details CSV found. Pass --match-csv or provide results root with by_run CSVs.")

    rows = _load_rows(match_csv)
    out = suggest(rows=rows, min_iou=args.min_iou, min_support=args.min_support)
    out["source_match_csv"] = str(match_csv.resolve())

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[OK] wrote {p}")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
