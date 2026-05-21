"""
s6_final_model.py — 최종 모델 선정 + 전체 학습 + 저장

s5의 model_comparison_results.csv에서 F2 기준 최적 모델 선정 후
전체 데이터로 학습하여 최종 모델 저장.

입력: data/ml/model_comparison_results.csv, model_fold_details.csv,
      best_config.json, v2_X/Y_{coin}.csv
출력: outputs/ml/final_model_{coin}.pkl, final_performance.csv, final_report.txt
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import ADASYN, SMOTE

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ML_DIR = os.path.join(PROJECT_DIR, "data", "ml")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ml")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42


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
            random_state=RANDOM_STATE, n_jobs=-1
        )
    elif model_name == "XGB":
        spw = (n_neg / n_pos) if (use_weight and n_pos and n_pos > 0) else 1
        return XGBClassifier(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
        )
    elif model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=use_weight,
            verbosity=-1, random_state=RANDOM_STATE, n_jobs=-1
        )
    elif model_name == "SVM":
        return SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, class_weight="balanced",
            random_state=RANDOM_STATE
        )


# ── 메인 로직 ──

def select_best_model(df_results, coin):
    sub = df_results[df_results["coin"] == coin].copy()
    sub = sub.sort_values(["f2_mean", "recall_mean", "f2_std"],
                          ascending=[False, False, True])
    return sub.iloc[0]


def train_final_model(coin, model_name, technique, threshold, feat_names):
    X, Y, _ = load_data(coin)

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    scaler = None
    if model_name == "SVM":
        scaler = RobustScaler()
        X_imp = scaler.fit_transform(X_imp)

    n_pos = int(Y.sum())
    sampler = build_sampler(technique, n_pos)
    if sampler is not None:
        X_train, Y_train = sampler.fit_resample(X_imp, Y)
    else:
        X_train, Y_train = X_imp, Y

    n_pos_res = int((Y_train == 1).sum())
    n_neg_res = int((Y_train == 0).sum())
    model = build_model(model_name, technique, n_pos=n_pos_res, n_neg=n_neg_res)
    model.fit(X_train, Y_train)

    # Resubstitution (참고용)
    y_prob = model.predict_proba(X_imp)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    resub = {
        "auc_roc": roc_auc_score(Y, y_prob),
        "auc_pr": average_precision_score(Y, y_prob),
        "recall": recall_score(Y, y_pred, zero_division=0),
        "precision": precision_score(Y, y_pred, zero_division=0),
        "f1": f1_score(Y, y_pred, zero_division=0),
        "f2": fbeta_score(Y, y_pred, beta=2.0),
    }

    return model, imp, scaler, X_imp, Y, y_pred, resub


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (coin, data) in zip(axes, results.items()):
        cm = confusion_matrix(data["Y"], data["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Depeg"])
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title(f"{coin} — Confusion Matrix\n"
                     f"({data['model_name']}, threshold={data['threshold']:.3f})")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "final_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path}")


def main():
    # 결과 로드
    df_results = pd.read_csv(os.path.join(ML_DIR, "model_comparison_results.csv"))
    df_folds = pd.read_csv(os.path.join(ML_DIR, "model_fold_details.csv"))

    with open(os.path.join(ML_DIR, "best_config.json"), "r", encoding="utf-8") as f:
        best_config = json.load(f)

    final_performance = []
    cm_results = {}
    report_lines = []

    report_lines.append("=" * 60)
    report_lines.append("  최종 모델 리포트")
    report_lines.append("=" * 60)

    for coin in ["USDC", "DAI"]:
        print(f"\n{'='*60}")
        print(f"  {coin} 최종 모델")
        print(f"{'='*60}")

        # 1. 최적 모델 선정
        best = select_best_model(df_results, coin)
        model_name = best["model"]
        technique = best["technique"]

        # 2. CV fold 평균 threshold
        fold_sub = df_folds[(df_folds["coin"] == coin) & (df_folds["model"] == model_name)]
        threshold = fold_sub["threshold"].mean()

        print(f"  선정: {technique}+{model_name}")
        print(f"  CV F2: {best['f2_mean']:.4f}±{best['f2_std']:.4f}")
        print(f"  CV Threshold (평균): {threshold:.4f}")

        # 3. 전체 학습
        _, _, feat_names = load_data(coin)
        model, imp, scaler, X_imp, Y, y_pred, resub = train_final_model(
            coin, model_name, technique, threshold, feat_names
        )

        print(f"  Resubstitution: F2={resub['f2']:.4f}, "
              f"Recall={resub['recall']:.4f}, Precision={resub['precision']:.4f}")

        # 4. 모델 저장
        save_obj = {
            "model": model,
            "imputer": imp,
            "scaler": scaler,
            "threshold": threshold,
            "feature_names": feat_names,
            "config": {
                "coin": coin,
                "model_name": model_name,
                "technique": technique,
            },
            "cv_performance": {
                "f2_mean": best["f2_mean"],
                "f2_std": best["f2_std"],
                "recall_mean": best["recall_mean"],
                "precision_mean": best["precision_mean"],
                "auc_pr_mean": best["auc_pr_mean"],
                "auc_roc_mean": best["auc_roc_mean"],
            }
        }

        pkl_path = os.path.join(OUT_DIR, f"final_model_{coin.lower()}.pkl")
        joblib.dump(save_obj, pkl_path)
        print(f"  저장: {pkl_path}")

        # 성능 기록
        perf = {
            "coin": coin,
            "model": model_name,
            "technique": technique,
            "threshold": round(threshold, 4),
            "cv_f2_mean": round(best["f2_mean"], 4),
            "cv_f2_std": round(best["f2_std"], 4),
            "cv_recall": round(best["recall_mean"], 4),
            "cv_precision": round(best["precision_mean"], 4),
            "cv_auc_pr": round(best["auc_pr_mean"], 4),
            "cv_auc_roc": round(best["auc_roc_mean"], 4),
            "resub_f2": round(resub["f2"], 4),
            "resub_recall": round(resub["recall"], 4),
            "resub_precision": round(resub["precision"], 4),
        }
        final_performance.append(perf)

        cm_results[coin] = {
            "Y": Y, "y_pred": y_pred,
            "model_name": model_name, "threshold": threshold
        }

        # 리포트
        report_lines.append(f"\n[{coin}]")
        report_lines.append(f"  모델: {technique}+{model_name}")
        report_lines.append(f"  Threshold: {threshold:.4f} (CV fold 평균)")
        report_lines.append(f"  CV 성능:")
        report_lines.append(f"    F2:        {best['f2_mean']:.4f} ± {best['f2_std']:.4f}")
        report_lines.append(f"    Recall:    {best['recall_mean']:.4f}")
        report_lines.append(f"    Precision: {best['precision_mean']:.4f}")
        report_lines.append(f"    AUC-PR:    {best['auc_pr_mean']:.4f}")
        report_lines.append(f"    AUC-ROC:   {best['auc_roc_mean']:.4f}")
        report_lines.append(f"  Resubstitution:")
        report_lines.append(f"    F2={resub['f2']:.4f}, Recall={resub['recall']:.4f}, "
                            f"Precision={resub['precision']:.4f}")

    # 성능 CSV
    df_perf = pd.DataFrame(final_performance)
    perf_path = os.path.join(OUT_DIR, "final_performance.csv")
    df_perf.to_csv(perf_path, index=False)
    print(f"\n성능 저장: {perf_path}")

    # 리포트 txt
    report_lines.append(f"\n{'='*60}")
    report_lines.append(f"  CV: StratifiedKFold(n_splits=5, shuffle=True)")
    report_lines.append(f"  한계: 시계열 순서 미반영 (data leakage 가능성)")
    report_lines.append(f"{'='*60}")

    report_path = os.path.join(OUT_DIR, "final_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"리포트 저장: {report_path}")

    # Confusion matrix
    print("\n시각화 생성 중...")
    plot_confusion_matrices(cm_results)

    # 검증: pkl 로드 테스트
    print(f"\n{'='*60}")
    print(f"  모델 로드 검증")
    print(f"{'='*60}")
    for coin in ["USDC", "DAI"]:
        pkl_path = os.path.join(OUT_DIR, f"final_model_{coin.lower()}.pkl")
        obj = joblib.load(pkl_path)
        m = obj["model"]
        X_test = np.random.randn(5, len(obj["feature_names"]))
        X_test = obj["imputer"].transform(X_test)
        if obj["scaler"] is not None:
            X_test = obj["scaler"].transform(X_test)
        proba = m.predict_proba(X_test)[:, 1]
        print(f"  [{coin}] 로드 OK, predict_proba shape={proba.shape}, "
              f"threshold={obj['threshold']:.4f}")

    print(f"\n{'='*60}")
    print(f"  최종 모델링 완료")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
