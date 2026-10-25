"""Generate report figures from Experiment 1 fusion CSV outputs.

Usage:
    python experiments/plot_experiment_1.py
    python experiments/plot_experiment_1.py --results-dir /path/to/run_folder
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Folder containing fusion_summary.csv and fusion_matched_pairs.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for PNG figures. Defaults to <results-dir>/graphs.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    default_results_dir = (
        project_root
        / "experiments"
        / "experiment_1_runs"
    )

    if args.results_dir is None:
        # If runs are stored in timestamped subfolders, choose the newest one.
        run_dirs = [
            p for p in default_results_dir.iterdir()
            if p.is_dir() and (p / "fusion_summary.csv").is_file()
        ] if default_results_dir.is_dir() else []
        if not run_dirs:
            raise FileNotFoundError(
                "Could not find a run folder containing fusion_summary.csv. "
                "Pass --results-dir with the path to your completed run folder."
            )
        results_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
    else:
        results_dir = args.results_dir.expanduser().resolve()

    summary_path = results_dir / "fusion_summary.csv"
    matches_path = results_dir / "fusion_matched_pairs.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing file: {summary_path}")
    if not matches_path.is_file():
        raise FileNotFoundError(f"Missing file: {matches_path}")

    summary = pd.read_csv(summary_path)
    matches = pd.read_csv(matches_path)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else results_dir / "graphs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    required_summary = {
        "frame_idx", "num_lidar_detections", "num_projected_boxes",
        "num_camera_candidates", "num_matches",
        "num_unprojected_boxes", "lidar_inference_seconds",
        "camera_inference_seconds", "total_seconds",
    }
    missing = required_summary - set(summary.columns)
    if missing:
        raise ValueError(f"fusion_summary.csv is missing columns: {sorted(missing)}")
    if "iou" not in matches.columns:
        raise ValueError("fusion_matched_pairs.csv must contain an 'iou' column.")

    # Figure 1: detection counts and successful projections across frames.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(summary["frame_idx"], summary["num_lidar_detections"],
            label="LiDAR detections", linewidth=1.5)
    ax.plot(summary["frame_idx"], summary["num_projected_boxes"],
            label="Projected LiDAR boxes", linewidth=1.5)
    ax.plot(summary["frame_idx"], summary["num_camera_candidates"],
            label="Camera candidates", linewidth=1.5)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Number of detections / boxes")
    ax.set_title("Experiment 1 — Detections and projection across frames")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_detections_projection.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: accepted matches in each frame.
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(summary["frame_idx"], summary["num_matches"], width=0.9)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Accepted camera–LiDAR matches")
    ax.set_title("Experiment 1 — Accepted fusion matches per frame")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_matches_per_frame.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: IoU distribution for accepted pairs.
    ious = matches["iou"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.hist(ious, bins=15, edgecolor="black")
    ax.axvline(0.30, linestyle="--", linewidth=1.5,
               label="Matching threshold = 0.30")
    if not ious.empty:
        ax.axvline(ious.mean(), linestyle=":", linewidth=1.8,
                   label=f"Mean matched IoU = {ious.mean():.3f}")
    ax.set_xlabel("2D bounding-box IoU")
    ax.set_ylabel("Number of matched pairs")
    ax.set_title(f"Experiment 1 — IoU distribution ({len(ious)} matched pairs)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_matched_iou_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: aggregate counts across all frames.
    totals = {
        "LiDAR detections": int(summary["num_lidar_detections"].sum()),
        "Projected LiDAR boxes": int(summary["num_projected_boxes"].sum()),
        "Unprojected LiDAR boxes": int(summary["num_unprojected_boxes"].sum()),
        "Camera candidates": int(summary["num_camera_candidates"].sum()),
        "Accepted matches": int(summary["num_matches"].sum()),
    }
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(list(totals.keys()), list(totals.values()))
    ax.set_ylabel("Total count across processed frames")
    ax.set_title("Experiment 1 — Aggregate detection and fusion counts")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, totals.values()):
        ax.annotate(f"{value:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_aggregate_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 5: inference and total runtime by frame.
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(summary["frame_idx"], summary["lidar_inference_seconds"],
            label="LiDAR inference", linewidth=1.4)
    ax.plot(summary["frame_idx"], summary["camera_inference_seconds"],
            label="Camera inference", linewidth=1.4)
    ax.plot(summary["frame_idx"], summary["total_seconds"],
            label="Total frame time", linewidth=1.4)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Experiment 1 — Runtime per frame")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig5_runtime_per_frame.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Read results from: {results_dir}")
    print(f"Saved five figures to: {output_dir}")
    for path in sorted(output_dir.glob("*.png")):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
