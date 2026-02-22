import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _try_import_seaborn():
    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid")
        return sns
    except Exception:
        return None


def plot_quality_tradeoff(df: pd.DataFrame, out_dir: Path, sns):
    d = df.copy()
    d = d.dropna(subset=["f1", "e2e_latency_ms"])
    if d.empty:
        return
    plt.figure(figsize=(8, 6))
    if sns is not None:
        ax = sns.scatterplot(
            data=d,
            x="e2e_latency_ms",
            y="f1",
            hue="scene",
            size="num_objects_in_map",
            sizes=(20, 200),
        )
    else:
        ax = plt.gca()
        ax.scatter(d["e2e_latency_ms"], d["f1"], s=40)
    ax.set_xlabel("End-to-end Latency (ms)")
    ax.set_ylabel("F1")
    ax.set_title("Quality vs Latency Tradeoff")
    plt.tight_layout()
    p = out_dir / "quality_vs_latency.png"
    plt.savefig(p, dpi=200)
    plt.close()


def plot_per_class_f1(df: pd.DataFrame, out_dir: Path, sns):
    d = df.copy()
    d = d.dropna(subset=["class", "f1", "run_id"])
    if d.empty:
        return
    plt.figure(figsize=(12, 6))
    if sns is not None:
        ax = sns.barplot(data=d, x="class", y="f1", hue="run_id")
    else:
        ax = plt.gca()
        # fallback: aggregate simple mean per class
        m = d.groupby("class", as_index=False)["f1"].mean()
        ax.bar(m["class"], m["f1"])
    ax.set_xlabel("Class")
    ax.set_ylabel("Per-class F1")
    ax.set_title("Per-class F1 by Run")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p = out_dir / "per_class_f1.png"
    plt.savefig(p, dpi=200)
    plt.close()


def plot_per_class_tpfpfn(df: pd.DataFrame, out_dir: Path, sns):
    d = df.copy()
    d = d.dropna(subset=["class", "tp", "fp", "fn"])
    if d.empty:
        return
    # Use first run by default for clean single-panel diagnostic.
    rid = sorted(d["run_id"].dropna().unique().tolist())[0]
    d = d[d["run_id"] == rid]
    melted = d.melt(id_vars=["class"], value_vars=["tp", "fp", "fn"], var_name="metric", value_name="count")
    plt.figure(figsize=(12, 6))
    if sns is not None:
        ax = sns.barplot(data=melted, x="class", y="count", hue="metric")
    else:
        ax = plt.gca()
        # fallback grouped bars
        classes = melted["class"].unique().tolist()
        width = 0.25
        xs = range(len(classes))
        for i, m in enumerate(["tp", "fp", "fn"]):
            vals = [float(melted[(melted["class"] == c) & (melted["metric"] == m)]["count"].sum()) for c in classes]
            ax.bar([x + (i - 1) * width for x in xs], vals, width=width, label=m)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.legend()
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-class TP/FP/FN ({rid})")
    plt.tight_layout()
    p = out_dir / "per_class_tpfpfn.png"
    plt.savefig(p, dpi=200)
    plt.close()


def plot_registration_retrieval_diagnostics(df: pd.DataFrame, out_dir: Path, sns):
    d = df.copy()
    needed = ["registration_helped_rate", "retrieval_swapped_tp_rate"]
    for c in needed:
        if c not in d.columns:
            return
    d = d.dropna(subset=needed)
    if d.empty:
        return
    plt.figure(figsize=(8, 6))
    if sns is not None:
        ax = sns.scatterplot(
            data=d,
            x="retrieval_swapped_tp_rate",
            y="registration_helped_rate",
            hue="scene" if "scene" in d.columns else None,
            size="f1" if "f1" in d.columns else None,
            sizes=(40, 220),
        )
    else:
        ax = plt.gca()
        ax.scatter(d["retrieval_swapped_tp_rate"], d["registration_helped_rate"], s=50)
    if "run_id" in d.columns:
        for _, r in d.iterrows():
            try:
                ax.annotate(str(r["run_id"]), (float(r["retrieval_swapped_tp_rate"]), float(r["registration_helped_rate"])), fontsize=7, alpha=0.7)
            except Exception:
                pass
    ax.set_xlabel("Retrieval Swap TP Rate")
    ax.set_ylabel("Registration Helped Rate")
    ax.set_title("Registration vs Retrieval Diagnostics")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    p = out_dir / "registration_vs_retrieval.png"
    plt.savefig(p, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate paper-friendly plots from evaluation tables.")
    default_results = str((Path(__file__).resolve().parent / "results").resolve())
    parser.add_argument("--results-root", default=default_results)
    parser.add_argument("--main-csv", default=None, help="Optional explicit ablation_main CSV")
    parser.add_argument("--per-class-csv", default=None, help="Optional explicit per_class CSV")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    tables_dir = results_root / "tables"
    plots_dir = results_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    main_csv = Path(args.main_csv) if args.main_csv else sorted(tables_dir.glob("ablation_main_*.csv"))[-1]
    per_class_csv = Path(args.per_class_csv) if args.per_class_csv else sorted(tables_dir.glob("per_class_*.csv"))[-1]

    main_df = pd.read_csv(main_csv)
    per_class_df = pd.read_csv(per_class_csv)
    sns = _try_import_seaborn()

    plot_quality_tradeoff(main_df, plots_dir, sns)
    plot_per_class_f1(per_class_df, plots_dir, sns)
    plot_per_class_tpfpfn(per_class_df, plots_dir, sns)
    plot_registration_retrieval_diagnostics(main_df, plots_dir, sns)
    print(f"[OK] wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
