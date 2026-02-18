"""
Refinement visualizations: uncertainty calibration, burst confusion matrix,
benchmark comparison. Reads pipeline outputs from output_dir, writes figures there.
"""

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12, "font.family": "sans-serif"})


def main(output_dir: Optional[str] = None) -> None:
    """Generate Figure4 (uncertainty), Figure5 (burst confusion), Figure6 (benchmark) in output_dir."""
    out = Path(output_dir) if output_dir else Path(".")
    preds_path = out / "all_predictions_and_uncertainty.csv"
    bench_path = out / "benchmark_results.csv"

    if not preds_path.exists():
        return
    preds = pd.read_csv(preds_path)
    df_n = preds[preds["Target"] == "Peppas_n"].copy()
    if not df_n.empty:
        df_n["AbsError"] = (df_n["Actual"] - df_n["Predicted"]).abs()
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=df_n, x="Uncertainty", y="AbsError",
            scatter_kws={"alpha": 0.3, "color": "blue"}, line_kws={"color": "red"},
        )
        corr = df_n["Uncertainty"].corr(df_n["AbsError"])
        plt.title(f"Uncertainty Calibration (Peppas n)\nCorrelation: {corr:.2f}")
        plt.xlabel("Ensemble Uncertainty (Std Dev)")
        plt.ylabel("Absolute Prediction Error")
        plt.tight_layout()
        plt.savefig(out / "Figure4_UncertaintyCalibration.png")
        plt.close()

    df_b = preds[preds["Target"] == "Burst_Class"].copy()
    if not df_b.empty:
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(df_b["Actual"], df_b["Predicted"])
        cmn = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        sns.heatmap(
            cmn, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Low (<10%)", "Med", "High (>40%)"],
            yticklabels=["Low", "Med", "High"],
        )
        plt.title("Burst Release Classification Accuracy")
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.tight_layout()
        plt.savefig(out / "Figure5_BurstClassification.png")
        plt.close()

    if bench_path.exists():
        bench = pd.read_csv(bench_path)
        bench = bench[bench["Target"].isin(["Peppas_n", "Peppas_K"])]
        if not bench.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=bench, x="Target", y="R2", hue="Model", palette="viridis")
            plt.axhline(0, color="k", linestyle="--", linewidth=1)
            plt.title("Rigorous Benchmarking: Stacked Ensemble vs Baselines")
            plt.ylabel("R2 Score (10-Fold CV)")
            plt.tight_layout()
            plt.savefig(out / "Figure6_Benchmarking.png")
            plt.close()


if __name__ == "__main__":
    try:
        import config as _cfg
        main(str(_cfg.OUTPUT_DIR))
    except ImportError:
        main(".")
