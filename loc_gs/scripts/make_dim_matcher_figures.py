#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = {
    "full_q20": {
        "label": "Full STDLoc parity",
        "summary": "output/stdloc_hybrid/ShopFacade_full_sota/eval_q20_blend08_dense2_r8/summary.json",
        "results": "output/stdloc_hybrid/ShopFacade_full_sota/eval_q20_blend08_dense2_r8/results.json",
        "color": "#4C78A8",
    },
    "parity_q20": {
        "label": "PLY parity",
        "summary": "output/stdloc_hybrid/ShopFacade_stdloc_parity/eval_q20_camerasjson_stdlocdet_original_plyloc_dense_parity_fixedcx_renderparity_syncK/summary.json",
        "results": "output/stdloc_hybrid/ShopFacade_stdloc_parity/eval_q20_camerasjson_stdlocdet_original_plyloc_dense_parity_fixedcx_renderparity_syncK/results.json",
        "color": "#59A14F",
    },
    "lg_loc": {
        "label": "LightGlue locability",
        "summary": "output/stdloc_hybrid/ShopFacade_dim_lightglue/eval_q20/summary.json",
        "results": "output/stdloc_hybrid/ShopFacade_dim_lightglue/eval_q20/results.json",
        "color": "#E15759",
    },
    "lg_detector": {
        "label": "LightGlue detector",
        "summary": "output/stdloc_hybrid/ShopFacade_dim_lightglue/eval_q20_detector/summary.json",
        "results": "output/stdloc_hybrid/ShopFacade_dim_lightglue/eval_q20_detector/results.json",
        "color": "#B07AA1",
    },
    "lg_k4096": {
        "label": "LightGlue 4096",
        "summary": "output/stdloc_hybrid/ShopFacade_dim_lightglue/eval_q20_k4096_f001/summary.json",
        "results": "output/stdloc_hybrid/ShopFacade_dim_lightglue/eval_q20_k4096_f001/results.json",
        "color": "#F28E2B",
    },
}


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def available_runs() -> list[str]:
    return [key for key, run in RUNS.items() if Path(run["summary"]).exists() and Path(run["results"]).exists()]


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[str]:
    paths = []
    for ext, dpi in (("pdf", None), ("svg", None), ("png", 300)):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_ablation(out_dir: Path, keys: list[str]) -> list[str]:
    labels = [RUNS[k]["label"] for k in keys]
    med = [load_json(RUNS[k]["summary"])["dense"]["median_te"] for k in keys]
    recall = [100.0 * load_json(RUNS[k]["summary"])["dense"]["recall_5cm_5d"] for k in keys]
    colors = [RUNS[k]["color"] for k in keys]
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1))
    axes[0].bar(x, med, color=colors)
    axes[0].axhline(2.647, color="#333333", linestyle="--", linewidth=1.1, label="STDLoc baseline")
    axes[0].set_ylabel("Median translation error (cm)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].legend(frameon=False, fontsize=8)
    style(axes[0])
    axes[1].bar(x, recall, color=colors)
    axes[1].axhline(80.58, color="#333333", linestyle="--", linewidth=1.1)
    axes[1].set_ylabel("R@5cm (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    style(axes[1])
    fig.suptitle("ShopFacade q20 matcher ablation", y=1.04, fontsize=12)
    return save_figure(fig, out_dir, "fig_dim_matcher_ablation_q20")


def plot_cdf(out_dir: Path, keys: list[str]) -> list[str]:
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for key in keys:
        rows = load_json(RUNS[key]["results"])
        errors = np.sort(np.array([row["dense_te"] for row in rows if row["dense_te"] is not None], dtype=np.float64))
        if errors.size == 0:
            continue
        y = np.arange(1, errors.size + 1) / errors.size
        ax.plot(errors, 100.0 * y, linewidth=2.0, label=RUNS[key]["label"], color=RUNS[key]["color"])
    ax.axvline(5.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Translation error (cm)")
    ax.set_ylabel("Localized queries (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    return save_figure(fig, out_dir, "fig_dim_matcher_cdf_q20")


def write_tail_table(out_dir: Path, keys: list[str]) -> str:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Med. t (cm) & R@5cm (\%) & Avg. inliers & Worst t (cm) \\",
        r"\midrule",
    ]
    for key in keys:
        summary = load_json(RUNS[key]["summary"])["dense"]
        rows = load_json(RUNS[key]["results"])
        worst = max([row["dense_te"] for row in rows if row["dense_te"] is not None] or [float("nan")])
        lines.append(
            f"{RUNS[key]['label']} & {summary['median_te']:.3f} & "
            f"{100.0 * summary['recall_5cm_5d']:.2f} & {summary['avg_inliers']:.1f} & {worst:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path = out_dir / "table_dim_matcher_tail_q20.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DIM matcher paper figures.")
    parser.add_argument("--output_dir", default="output/paper_figures/dim_matchers/ShopFacade")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    keys = available_runs()
    artifacts: list[str] = []
    artifacts.extend(plot_ablation(out_dir, keys))
    artifacts.extend(plot_cdf(out_dir, keys))
    artifacts.append(write_tail_table(out_dir, keys))
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": artifacts}, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "artifacts": artifacts}, indent=2))


if __name__ == "__main__":
    main()
