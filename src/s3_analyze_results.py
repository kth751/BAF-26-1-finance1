"""
s3_analyze_results.py — 불균형 실험 결과 분석 + 최적 기법 추천 + 시각화

입력: data/ml/imbalance_results.csv (StratifiedKFold 5-fold, macro-average)
출력: data/ml/best_config.json, outputs/ml/imbalance_*.png
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ML_DIR = os.path.join(PROJECT_DIR, "data", "ml")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ml")
os.makedirs(OUT_DIR, exist_ok=True)


def load_results():
    return pd.read_csv(os.path.join(ML_DIR, "imbalance_results.csv"))


def print_summary_table(df):
    print("\n" + "=" * 100)
    print("  StratifiedKFold 5-Fold 결과 (F2 기준 정렬)")
    print("=" * 100)

    for coin in df["coin"].unique():
        sub = df[df["coin"] == coin].copy()
        sub = sub.sort_values("f2_mean", ascending=False)
        print(f"\n--- {coin} (디페깅 {sub.iloc[0].get('n_pos_test', '?')}건/fold) ---")
        print(f"{'기법+모델':<30s} {'F2':>10s} {'Recall':>10s} {'Precision':>12s} {'AUC-PR':>10s} {'AUC-ROC':>10s}")
        for _, r in sub.iterrows():
            label = f"{r['technique']}+{r['model']}"
            print(f"{label:<30s} "
                  f"{r['f2_mean']:>.3f}±{r['f2_std']:>.3f} "
                  f"{r['recall_mean']:>.3f}±{r['recall_std']:>.3f} "
                  f"{r['precision_mean']:>.3f}±{r['precision_std']:>.3f} "
                  f"{r['auc_pr_mean']:>.3f}±{r['auc_pr_std']:>.3f} "
                  f"{r['auc_roc_mean']:>.3f}±{r['auc_roc_std']:>.3f}")


def recommend_best(df):
    best = {}
    for coin in df["coin"].unique():
        sub = df[df["coin"] == coin].copy()

        # F2 기준 (조기경보 → Recall 가중)
        f2_best = sub.sort_values(["f2_mean", "recall_mean"], ascending=False).iloc[0]

        # Recall 기준 (탐지 최우선)
        recall_best = sub.sort_values(["recall_mean", "f2_mean"], ascending=False).iloc[0]

        # AUC-PR 기준 (threshold 무관 랭킹 성능)
        auc_best = sub.sort_values("auc_pr_mean", ascending=False).iloc[0]

        # F1 기준 (Precision-Recall 균형)
        f1_best = sub.sort_values(["f1_mean", "precision_mean"], ascending=False).iloc[0]

        best[coin] = {
            "best_f2": {
                "technique": f2_best["technique"], "model": f2_best["model"],
                "f2": round(float(f2_best["f2_mean"]), 4),
                "f2_std": round(float(f2_best["f2_std"]), 4),
                "recall": round(float(f2_best["recall_mean"]), 4),
                "precision": round(float(f2_best["precision_mean"]), 4),
                "auc_pr": round(float(f2_best["auc_pr_mean"]), 4),
            },
            "best_recall": {
                "technique": recall_best["technique"], "model": recall_best["model"],
                "recall": round(float(recall_best["recall_mean"]), 4),
                "f2": round(float(recall_best["f2_mean"]), 4),
                "precision": round(float(recall_best["precision_mean"]), 4),
            },
            "best_auc_pr": {
                "technique": auc_best["technique"], "model": auc_best["model"],
                "auc_pr": round(float(auc_best["auc_pr_mean"]), 4),
            },
            "best_f1": {
                "technique": f1_best["technique"], "model": f1_best["model"],
                "f1": round(float(f1_best["f1_mean"]), 4),
                "precision": round(float(f1_best["precision_mean"]), 4),
                "recall": round(float(f1_best["recall_mean"]), 4),
            },
        }

    out_path = os.path.join(ML_DIR, "best_config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print(f"\n최적 설정 저장: {out_path}")

    for coin, info in best.items():
        print(f"\n  [{coin}]")
        b = info["best_f2"]
        print(f"    F2 최고:     {b['technique']}+{b['model']} → F2={b['f2']}±{b['f2_std']}, R={b['recall']}, P={b['precision']}")
        b = info["best_recall"]
        print(f"    Recall 최고: {b['technique']}+{b['model']} → Recall={b['recall']}, F2={b['f2']}, P={b['precision']}")
        b = info["best_auc_pr"]
        print(f"    AUC-PR 최고: {b['technique']}+{b['model']} → AUC-PR={b['auc_pr']}")
        b = info["best_f1"]
        print(f"    F1 최고:     {b['technique']}+{b['model']} → F1={b['f1']}, P={b['precision']}, R={b['recall']}")

    return best


def plot_f2_barh(df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    for ax, coin in zip(axes, df["coin"].unique()):
        sub = df[df["coin"] == coin].copy()
        sub["label"] = sub["technique"] + "+" + sub["model"]
        sub = sub.sort_values("f2_mean", ascending=True)

        colors = []
        for _, r in sub.iterrows():
            if "BalancedRF" in r["model"] or "EasyEnsemble" in r["model"]:
                colors.append("#55A868")
            elif "XGB" in r["model"]:
                colors.append("#DD8452")
            else:
                colors.append("#4C72B0")

        y = range(len(sub))
        ax.barh(y, sub["f2_mean"], xerr=sub["f2_std"], color=colors, alpha=0.85, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["label"], fontsize=8)
        ax.set_xlabel("F2 Score (mean ± std)")
        ax.set_title(f"{coin} — F2 Score by Technique (StratifiedKFold 5-Fold)")
        ax.set_xlim(0, 1.1)
        ax.axvline(x=sub["f2_mean"].max(), color="red", ls="--", lw=0.8, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "imbalance_f2_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path}")


def plot_metric_heatmap(df):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    metrics = ["f2_mean", "recall_mean", "precision_mean", "f1_mean", "auc_pr_mean", "auc_roc_mean"]
    labels = ["F2", "Recall", "Precision", "F1", "AUC-PR", "AUC-ROC"]

    for ax, coin in zip(axes, df["coin"].unique()):
        sub = df[df["coin"] == coin].copy()
        sub["label"] = sub["technique"] + "+" + sub["model"]
        sub = sub.sort_values("f2_mean", ascending=False)
        pivot = sub.set_index("label")[metrics]
        pivot.columns = labels

        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                    ax=ax, vmin=0, vmax=1, linewidths=0.5)
        ax.set_title(f"{coin} — Metrics Heatmap")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "imbalance_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path}")


def plot_recall_precision_tradeoff(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, coin in zip(axes, df["coin"].unique()):
        sub = df[df["coin"] == coin].copy()
        sub["label"] = sub["technique"] + "+" + sub["model"]

        colors = []
        for _, r in sub.iterrows():
            if "BalancedRF" in r["model"] or "EasyEnsemble" in r["model"]:
                colors.append("#55A868")
            elif "XGB" in r["model"]:
                colors.append("#DD8452")
            else:
                colors.append("#4C72B0")

        ax.scatter(sub["recall_mean"], sub["precision_mean"], c=colors, s=80, alpha=0.8, edgecolors="white")

        for _, r in sub.iterrows():
            ax.annotate(r["label"], (r["recall_mean"], r["precision_mean"]),
                       fontsize=6, alpha=0.7, ha="center", va="bottom")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{coin} — Precision-Recall Tradeoff")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "imbalance_pr_tradeoff.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path}")


def main():
    df = load_results()

    print_summary_table(df)
    best = recommend_best(df)

    print("\n시각화 생성 중...")
    plot_f2_barh(df)
    plot_metric_heatmap(df)
    plot_recall_precision_tradeoff(df)

    print("\n" + "=" * 100)
    print("  분석 완료")
    print("=" * 100)


if __name__ == "__main__":
    main()
