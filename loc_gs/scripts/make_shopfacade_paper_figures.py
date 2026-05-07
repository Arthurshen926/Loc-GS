#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


RUNS = {
    "parity_ply": {
        "label": "STDLoc parity + PLY",
        "summary": "output/stdloc_hybrid/ShopFacade_stdloc_parity/eval_full_camerasjson_stdlocdet_original_plyloc_dense2_syncK_r8/summary.json",
        "results": "output/stdloc_hybrid/ShopFacade_stdloc_parity/eval_full_camerasjson_stdlocdet_original_plyloc_dense2_syncK_r8/results.json",
        "color": "#4C78A8",
    },
    "xview": {
        "label": "+ Cross-view",
        "summary": "output/stdloc_hybrid/ShopFacade_xview/eval_full_blend09_dense2_r8/summary.json",
        "results": "output/stdloc_hybrid/ShopFacade_xview/eval_full_blend09_dense2_r8/results.json",
        "color": "#59A14F",
    },
    "hardneg": {
        "label": "+ Hard negatives",
        "summary": "output/stdloc_hybrid/ShopFacade_hardneg/eval_full_blend09_dense2_r8/summary.json",
        "results": "output/stdloc_hybrid/ShopFacade_hardneg/eval_full_blend09_dense2_r8/results.json",
        "color": "#F28E2B",
    },
    "splatloc": {
        "label": "+ Implicit saliency",
        "summary": "output/stdloc_hybrid/ShopFacade_splatloc_implicit/eval_full_blend09_dense2_r8/summary.json",
        "results": "output/stdloc_hybrid/ShopFacade_splatloc_implicit/eval_full_blend09_dense2_r8/results.json",
        "color": "#B07AA1",
    },
    "full": {
        "label": "Full",
        "summary": "output/stdloc_hybrid/ShopFacade_full_sota/eval_full_blend0.9_dense2_r8/summary.json",
        "results": "output/stdloc_hybrid/ShopFacade_full_sota/eval_full_blend0.9_dense2_r8/results.json",
        "color": "#E15759",
    },
    "full_ply": {
        "label": "Full ckpt, PLY only",
        "summary": "output/stdloc_hybrid/ShopFacade_full_sota/eval_full_plyloc_dense2_r8/summary.json",
        "results": "output/stdloc_hybrid/ShopFacade_full_sota/eval_full_plyloc_dense2_r8/results.json",
        "color": "#79706E",
    },
}

TRAIN_LOGS = {
    "xview": "output/stdloc_hybrid/ShopFacade_xview/origteacher_e2_nocache/metrics.jsonl",
    "hardneg": "output/stdloc_hybrid/ShopFacade_hardneg/origteacher_e2_nocache/metrics.jsonl",
    "splatloc": "output/stdloc_hybrid/ShopFacade_splatloc_implicit/origteacher_e2_nocache/metrics.jsonl",
    "full": "output/stdloc_hybrid/ShopFacade_full_sota/origteacher_e2_nocache/metrics.jsonl",
}


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dense_summary(run_key: str) -> dict:
    return load_json(RUNS[run_key]["summary"])["dense"]


def dense_results(run_key: str) -> list[dict]:
    return load_json(RUNS[run_key]["results"])


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[str]:
    out_paths = []
    for ext, dpi in (("pdf", None), ("svg", None), ("png", 300)):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        out_paths.append(str(path))
    plt.close(fig)
    return out_paths


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_ablation_bars(out_dir: Path) -> list[str]:
    keys = ["parity_ply", "xview", "hardneg", "splatloc", "full"]
    labels = [RUNS[k]["label"] for k in keys]
    colors = [RUNS[k]["color"] for k in keys]
    med = [dense_summary(k)["median_te"] for k in keys]
    recall = [100.0 * dense_summary(k)["recall_5cm_5d"] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1))
    x = np.arange(len(keys))
    axes[0].bar(x, med, color=colors, width=0.68)
    axes[0].axhline(2.647, color="#333333", linestyle="--", linewidth=1.2, label="STDLoc baseline")
    axes[0].set_ylabel("Median translation error (cm)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].legend(frameon=False, fontsize=8)
    style_axes(axes[0])

    axes[1].bar(x, recall, color=colors, width=0.68)
    axes[1].axhline(80.58, color="#333333", linestyle="--", linewidth=1.2, label="STDLoc baseline")
    axes[1].set_ylabel("5 cm recall (%)")
    axes[1].set_ylim(68, 84)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    style_axes(axes[1])

    fig.suptitle("ShopFacade localization ablation", y=1.03, fontsize=12)
    return save_figure(fig, out_dir, "fig_shopfacade_ablation_bars")


def plot_error_cdf(out_dir: Path) -> list[str]:
    keys = ["parity_ply", "hardneg", "full", "full_ply"]
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for key in keys:
        errors = np.array([row["dense_te"] for row in dense_results(key)], dtype=np.float64)
        errors = np.sort(errors)
        y = np.arange(1, len(errors) + 1, dtype=np.float64) / len(errors)
        ax.plot(errors, 100.0 * y, linewidth=2.0, label=RUNS[key]["label"], color=RUNS[key]["color"])
    ax.axvline(5.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Translation error (cm)")
    ax.set_ylabel("Localized queries (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax)
    return save_figure(fig, out_dir, "fig_shopfacade_translation_cdf")


def plot_paired_delta(out_dir: Path) -> list[str]:
    base = dense_results("full_ply")
    improved = dense_results("full")
    base_by_name = {row["image_name"]: row for row in base}
    deltas = []
    for row in improved:
        name = row["image_name"]
        if name not in base_by_name:
            continue
        deltas.append(base_by_name[name]["dense_te"] - row["dense_te"])
    deltas = np.asarray(deltas, dtype=np.float64)
    order = np.argsort(deltas)
    clip_min, clip_max = -10.0, 10.0
    clipped = np.clip(deltas[order], clip_min, clip_max)
    below = int((deltas < clip_min).sum())
    above = int((deltas > clip_max).sum())

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    colors = np.where(deltas[order] >= 0.0, "#59A14F", "#E15759")
    ax.bar(np.arange(len(deltas)), clipped, color=colors, width=0.9)
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    if below:
        ax.scatter(
            np.where(deltas[order] < clip_min)[0],
            np.full(below, clip_min),
            marker="v",
            s=22,
            color="#E15759",
            zorder=4,
        )
    if above:
        ax.scatter(
            np.where(deltas[order] > clip_max)[0],
            np.full(above, clip_max),
            marker="^",
            s=22,
            color="#59A14F",
            zorder=4,
        )
    ax.set_ylim(clip_min - 1.0, clip_max + 1.0)
    ax.set_xlabel("Test query sorted by gain")
    ax.set_ylabel("PLY-only error - Full error (cm)")
    ax.set_title("Per-query gain from hybrid residual descriptors")
    note = f"Clipped to [{clip_min:.0f}, {clip_max:.0f}] cm"
    if below or above:
        note += f"; {below} below, {above} above"
    ax.text(
        0.99,
        0.04,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    style_axes(ax)
    return save_figure(fig, out_dir, "fig_shopfacade_per_query_gain")


def plot_error_scatter(out_dir: Path) -> list[str]:
    keys = ["parity_ply", "hardneg", "full"]
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for key in keys:
        rows = dense_results(key)
        te = np.array([row["dense_te"] for row in rows], dtype=np.float64)
        ae = np.array([row["dense_ae"] for row in rows], dtype=np.float64)
        ax.scatter(te, ae, s=16, alpha=0.65, label=RUNS[key]["label"], color=RUNS[key]["color"], edgecolors="none")
    ax.axvline(5.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("Translation error (cm)")
    ax.set_ylabel("Rotation error (deg)")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    return save_figure(fig, out_dir, "fig_shopfacade_error_scatter")


def plot_training_curves(out_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), sharex=True)
    metrics = [("loss", "Training loss"), ("same_top1_1px", "Same-view top-1 @1px"), ("xview", "Cross-view loss")]
    for key, log_path in TRAIN_LOGS.items():
        rows = load_jsonl(log_path)
        if not rows:
            continue
        epochs = np.array([row["epoch"] for row in rows], dtype=np.float64)
        for ax, (metric, title) in zip(axes, metrics):
            values = np.array([row.get(metric, 0.0) for row in rows], dtype=np.float64)
            ax.plot(epochs, values, marker="o", linewidth=1.8, label=RUNS.get(key, {"label": key})["label"], color=RUNS.get(key, {"color": None})["color"])
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Epoch")
            style_axes(ax)
    axes[0].set_ylabel("Value")
    axes[-1].legend(frameon=False, fontsize=7, loc="best")
    return save_figure(fig, out_dir, "fig_shopfacade_training_curves")


def write_latex_table(out_dir: Path) -> str:
    keys = ["parity_ply", "xview", "hardneg", "splatloc", "full", "full_ply"]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Med. t (cm) & Med. R ($^\circ$) & R@5cm (\%) & R@2cm (\%) \\",
        r"\midrule",
    ]
    for key in keys:
        d = dense_summary(key)
        lines.append(
            f"{RUNS[key]['label']} & {d['median_te']:.3f} & {d['median_ae']:.3f} & "
            f"{100.0 * d['recall_5cm_5d']:.2f} & {100.0 * d['recall_2cm_2d']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = out_dir / "table_shopfacade_ablation.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready ShopFacade result figures.")
    parser.add_argument("--output_dir", default="output/paper_figures/shopfacade")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    artifacts: list[str] = []
    for plot_fn in (
        plot_ablation_bars,
        plot_error_cdf,
        plot_paired_delta,
        plot_error_scatter,
        plot_training_curves,
    ):
        artifacts.extend(plot_fn(out_dir))
    artifacts.append(write_latex_table(out_dir))

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": artifacts}, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "artifacts": artifacts}, indent=2))


if __name__ == "__main__":
    main()
