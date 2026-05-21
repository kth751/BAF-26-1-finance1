"""
s5_model_comparison.py — 추가 모델(LightGBM, SVM) + 기존(RF, XGB) 통합 비교

코인별 best technique 적용 + StratifiedKFold(5) + 4개 모델 비교.

입력: data/ml/best_config.json, v2_X/Y_{coin}.csv
출력: data/ml/model_comparison_results.csv, model_fold_details.csv,
      outputs/ml/model_comparison_*.png
"""

import pandas as pd
import numpy as np
import os
import json
import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score, recall_score, precision_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import ADASYN, SMOTE

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ML_DIR = os.path.join(PROJECT_DIR, "data", "ml")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ml")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5


# ── 재사용 함수 ──

def load_data(coin):
    X = pd.read_csv(os.path.join(ML_DIR, f"v2_X_{coin.lower()}.csv"))
    Y = pd.read_csv(os.path.join(ML_DIR, f"v2_Y_{coin.lower()}.csv"), header=None).squeeze()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.values, Y.values, X.columns.tolist()


def fbeta_score(y_true, y_pred, beta):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    if p + r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r)


def find_best_threshold(y_true, y_prob, beta=2.0):
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    eps = 1e-8
    fb = ((1 + beta**2) * prec[:-1] * rec[:-1]
          / (beta**2 * prec[:-1] + rec[:-1] + eps))
    best_idx = fb.argmax()
    return thresholds[best_idx]


def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        return {k: np.nan for k in ["auc_roc", "auc_pr", "recall", "precision", "f1", "f2"]}
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2.0),
    }


def build_sampler(technique, n_minority, ratio=0.5):
    if technique is None or technique in ("WeightOnly", "Baseline"):
        return None
    k = min(5, n_minority - 1)
    if k < 1:
        return None
    if technique == "ADASYN":
        return ADASYN(sampling_strategy=ratio, n_neighbors=k, random_state=RANDOM_STATE)
    elif technique.startswith("SMOTE"):
        return SMOTE(sampling_strategy=ratio, k_neighbors=k, random_state=RANDOM_STATE)
    return None


def build_model(model_name, technique, n_pos=None, n_neg=None):
    use_weight = (technique == "WeightOnly")

    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=300, max_depth=8,
            class_weight="balanced" if use_weight else None,
            random_state=RANDOM_STATE, n_jobs=1
        )
    elif model_name == "XGB":
        spw = (n_neg / n_pos) if (use_weight and n_pos and n_pos > 0) else 1
        return XGBClassifier(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0, n_jobs=1
        )
    elif model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=use_weight,
            verbosity=-1, random_state=RANDOM_STATE, n_jobs=1
        )
    elif model_name == "SVM":
        return SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, class_weight="balanced",
            random_state=RANDOM_STATE
        )


# ── 실험 실행 ──

def run_experiment(X, Y, model_name, technique, skf):
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        n_pos = int(Y_train.sum())
        n_neg = int((Y_train == 0).sum())

        if n_pos < 2:
            continue

        # Impute
        imp = SimpleImputer(strategy="median")
        X_train = imp.fit_transform(X_train)
        X_test = imp.transform(X_test)

        # SVM: RobustScaler (median/IQR 기반 — 이상치에 강건)
        scaler = None
        if model_name == "SVM":
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Sampler (train에만)
        sampler = build_sampler(technique, n_pos)
        if sampler is not None:
            try:
                X_train, Y_train = sampler.fit_resample(X_train, Y_train)
            except Exception:
                continue

        # Model
        n_pos_res = int((Y_train == 1).sum())
        n_neg_res = int((Y_train == 0).sum())
        model = build_model(model_name, technique, n_pos=n_pos_res, n_neg=n_neg_res)
        model.fit(X_train, Y_train)

        y_prob = model.predict_proba(X_test)[:, 1]

        # s2와 동일하게 threshold=0.5 기본값 사용
        threshold = 0.5

        # 최적 threshold도 참고용으로 기록
        y_prob_train = model.predict_proba(
            scaler.transform(imp.transform(X[train_idx])) if scaler else imp.transform(X[train_idx])
        )[:, 1]
        opt_threshold = find_best_threshold(Y[train_idx], y_prob_train, beta=2.0)

        metrics = evaluate(Y_test, y_prob, threshold=threshold)
        metrics["fold"] = fold_idx
        metrics["threshold"] = threshold
        metrics["opt_threshold"] = opt_threshold
        fold_metrics.append(metrics)

    if not fold_metrics:
        return None, None

    df_folds = pd.DataFrame(fold_metrics)

    # Summary
    summary = {}
    for metric in ["auc_roc", "auc_pr", "recall", "precision", "f1", "f2"]:
        vals = df_folds[metric].dropna()
        summary[f"{metric}_mean"] = vals.mean()
        summary[f"{metric}_std"] = vals.std()
    summary["n_valid_folds"] = len(fold_metrics)
    summary["avg_threshold"] = df_folds["threshold"].mean()

    return summary, df_folds


# ── 시각화 ──

def plot_f2_comparison(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, coin in zip(axes, df["coin"].unique()):
        sub = df[df["coin"] == coin].sort_values("f2_mean", ascending=True)
        colors = []
        for m in sub["model"]:
            if m == "XGB":
                colors.append("#DD8452")
            elif m == "LightGBM":
                colors.append("#55A868")
            elif m == "SVM":
                colors.append("#C44E52")
            else:
                colors.append("#4C72B0")

        y = range(len(sub))
        ax.barh(y, sub["f2_mean"], xerr=sub["f2_std"], color=colors, alpha=0.85, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['technique']}+{r['model']}" for _, r in sub.iterrows()], fontsize=9)
        ax.set_xlabel("F2 Score (mean ± std)")
        ax.set_title(f"{coin} — Model Comparison (F2)")
        ax.set_xlim(0, 1.1)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "model_comparison_f2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path}")


def plot_heatmap(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
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

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "model_comparison_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path}")


def plot_boxplot(fold_details):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, coin in zip(axes, fold_details["coin"].unique()):
        sub = fold_details[fold_details["coin"] == coin]
        models = sub["model"].unique()

        data = [sub[sub["model"] == m]["f2"].dropna().values for m in models]
        labels = [f"{sub[sub['model']==m]['technique'].iloc[0]}+{m}" for m in models]

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = {"RF": "#4C72B0", "XGB": "#DD8452", "LightGBM": "#55A868", "SVM": "#C44E52"}
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(colors.get(m, "#999999"))
            patch.set_alpha(0.7)

        ax.set_ylabel("F2 Score")
        ax.set_title(f"{coin} — F2 by Fold (Boxplot)")
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "model_comparison_boxplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path}")


# ── 메인 ──

def main():
    with open(os.path.join(ML_DIR, "best_config.json"), "r", encoding="utf-8") as f:
        best_config = json.load(f)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = ["RF", "XGB", "LightGBM", "SVM"]

    all_results = []
    all_fold_details = []

    for coin in ["USDC", "DAI"]:
        technique = best_config[coin]["best_f2"]["technique"]

        print(f"\n{'='*60}")
        print(f"  {coin} 모델 비교 (technique={technique})")
        print(f"  CV: StratifiedKFold(n_splits={N_SPLITS}, shuffle=True)")
        print(f"{'='*60}")

        X, Y, feat_names = load_data(coin)
        print(f"  데이터: {X.shape}, 디페깅 비율: {Y.mean()*100:.1f}% ({int(Y.sum())}건)\n")

        for model_name in models:
            t0 = time.time()
            label = f"{technique}+{model_name}"
            print(f"  {label:30s}", end=" ", flush=True)

            summary, df_folds = run_experiment(X, Y, model_name, technique, skf)
            elapsed = time.time() - t0

            if summary is None:
                print(f"→ FAILED ({elapsed:.1f}s)")
                continue

            summary["coin"] = coin
            summary["technique"] = technique
            summary["model"] = model_name
            all_results.append(summary)

            # fold 상세 저장
            df_folds["coin"] = coin
            df_folds["technique"] = technique
            df_folds["model"] = model_name
            all_fold_details.append(df_folds)

            print(f"→ F2={summary['f2_mean']:.3f}±{summary['f2_std']:.3f}  "
                  f"Recall={summary['recall_mean']:.3f}  "
                  f"Prec={summary['precision_mean']:.3f}  "
                  f"AUC-PR={summary['auc_pr_mean']:.3f}  ({elapsed:.1f}s)")

    # 결과 저장
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(ML_DIR, "model_comparison_results.csv"), index=False)
    print(f"\n결과 저장: model_comparison_results.csv ({len(df_results)}행)")

    df_fold_all = pd.concat(all_fold_details, ignore_index=True)
    df_fold_all.to_csv(os.path.join(ML_DIR, "model_fold_details.csv"), index=False)
    print(f"Fold 상세 저장: model_fold_details.csv ({len(df_fold_all)}행)")

    # 시각화
    print("\n시각화 생성 중...")
    plot_f2_comparison(df_results)
    plot_heatmap(df_results)
    plot_boxplot(df_fold_all)

    # 요약
    print(f"\n{'='*60}")
    print(f"  코인별 최적 모델 (F2 기준)")
    print(f"{'='*60}")
    for coin in ["USDC", "DAI"]:
        sub = df_results[df_results["coin"] == coin].sort_values("f2_mean", ascending=False)
        best = sub.iloc[0]
        print(f"  [{coin}] {best['technique']}+{best['model']} "
              f"→ F2={best['f2_mean']:.4f}±{best['f2_std']:.4f}, "
              f"Recall={best['recall_mean']:.4f}, Precision={best['precision_mean']:.4f}")


if __name__ == "__main__":
    main()
