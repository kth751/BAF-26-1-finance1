"""
s8_retrain_v3.py — v3 데이터(텍스트 변수 포함) 모델 재훈련

v2 최적 설정 그대로 적용 + v2 vs v3 성능 비교
- USDC: WeightOnly + XGB
- DAI: ADASYN + XGB
- CV: StratifiedKFold(5, shuffle=True)

추가: 전체 불균형 기법 비교도 실행하여 v3에서 최적 기법이 바뀌는지 확인
"""

import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ML_DIR = os.path.join(PROJECT_DIR, "data", "ml")

RANDOM_STATE = 42
N_SPLITS = 5


def load_data(coin, version="v3"):
    X = pd.read_csv(os.path.join(ML_DIR, f"{version}_X_{coin.lower()}.csv"))
    Y = pd.read_csv(os.path.join(ML_DIR, f"{version}_Y_{coin.lower()}.csv"), header=None).squeeze()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.values, Y.values, X.columns.tolist()


def fbeta(y_true, y_pred, beta=2.0):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    if p + r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r)


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
        "f2": fbeta(y_true, y_pred, beta=2.0),
    }


def run_experiment(X, Y, config, skf):
    fold_metrics = []
    skipped = 0

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        n_pos = int(Y_train.sum())
        n_neg = int((Y_train == 0).sum())

        if n_pos < 2:
            skipped += 1
            continue

        imp = SimpleImputer(strategy="median")
        X_train = imp.fit_transform(X_train)
        X_test = imp.transform(X_test)

        # Sampler
        if config["sampler"] == "ADASYN":
            k = min(5, n_pos - 1)
            if k < 1:
                skipped += 1
                continue
            try:
                sampler = ADASYN(sampling_strategy=0.5, n_neighbors=k, random_state=RANDOM_STATE)
                X_train, Y_train = sampler.fit_resample(X_train, Y_train)
            except Exception:
                skipped += 1
                continue
        elif config["sampler"] == "SMOTE":
            k = min(5, n_pos - 1)
            if k < 1:
                skipped += 1
                continue
            try:
                sampler = SMOTE(sampling_strategy=0.5, k_neighbors=k, random_state=RANDOM_STATE)
                X_train, Y_train = sampler.fit_resample(X_train, Y_train)
            except Exception:
                skipped += 1
                continue

        n_pos_r = int((Y_train == 1).sum())
        n_neg_r = int((Y_train == 0).sum())

        # Model
        if config["model"] == "RF":
            model = RandomForestClassifier(
                n_estimators=300, max_depth=8,
                class_weight="balanced" if config["weight"] else None,
                random_state=RANDOM_STATE, n_jobs=1
            )
        else:
            spw = (n_neg_r / n_pos_r) if config["weight"] and n_pos_r > 0 else 1
            model = XGBClassifier(
                n_estimators=300, max_depth=6,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=spw, eval_metric="logloss",
                random_state=RANDOM_STATE, verbosity=0, n_jobs=1
            )

        model.fit(X_train, Y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(Y_test, y_prob)
        metrics["fold"] = fold_idx
        metrics["n_pos_test"] = int(Y_test.sum())
        fold_metrics.append(metrics)

    if not fold_metrics:
        return None

    df_f = pd.DataFrame(fold_metrics)
    result = {}
    for m in ["auc_roc", "auc_pr", "recall", "precision", "f1", "f2"]:
        vals = df_f[m].dropna()
        result[f"{m}_mean"] = vals.mean()
        result[f"{m}_std"] = vals.std()
    result["n_valid_folds"] = len(fold_metrics)
    return result


def main():
    configs = [
        {"name": "WeightOnly", "model": "XGB", "sampler": None, "weight": True},
        {"name": "ADASYN", "model": "XGB", "sampler": "ADASYN", "weight": False},
        {"name": "SMOTE_05", "model": "RF", "sampler": "SMOTE", "weight": False},
        {"name": "Baseline", "model": "RF", "sampler": None, "weight": False},
        {"name": "Baseline", "model": "XGB", "sampler": None, "weight": False},
        {"name": "WeightOnly", "model": "RF", "sampler": None, "weight": True},
        {"name": "ADASYN", "model": "RF", "sampler": "ADASYN", "weight": False},
    ]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_results = []

    # v2 결과 로드
    v2_res = pd.read_csv(os.path.join(ML_DIR, "imbalance_results.csv"))

    for coin in ["USDC", "DAI"]:
        X, Y, feat_names = load_data(coin, "v3")

        print("=" * 65)
        print(f"  {coin} - v3 모델 재훈련 (텍스트 변수 포함)")
        print(f"  Data: {X.shape}, depeg: {int(Y.sum())} ({Y.mean()*100:.1f}%)")
        print("=" * 65)

        for cfg in configs:
            label = f"{cfg['name']}+{cfg['model']}"
            t0 = time.time()

            result = run_experiment(X, Y, cfg, skf)
            elapsed = time.time() - t0

            if result is None:
                print(f"  {label:30s} -> FAILED ({elapsed:.1f}s)")
                continue

            result["coin"] = coin
            result["technique"] = cfg["name"]
            result["model"] = cfg["model"]
            all_results.append(result)

            print(f"  {label:30s} -> F2={result['f2_mean']:.3f}+/-{result['f2_std']:.3f}  "
                  f"Recall={result['recall_mean']:.3f}  Prec={result['precision_mean']:.3f}  "
                  f"AUC-PR={result['auc_pr_mean']:.3f}  ({elapsed:.1f}s)")

    # === v2 vs v3 비교 ===
    v3_df = pd.DataFrame(all_results)

    print("\n" + "=" * 65)
    print("  v2 vs v3 성능 비교")
    print("=" * 65)

    for coin in ["USDC", "DAI"]:
        print(f"\n  --- {coin} ---")
        print(f"  {'기법':30s} | {'v2 F2':>8s} | {'v3 F2':>8s} | {'차이':>8s} | {'v2 AUC-PR':>9s} | {'v3 AUC-PR':>9s}")
        print(f"  {'-'*30}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}")

        for _, v3r in v3_df[v3_df["coin"] == coin].iterrows():
            tech = v3r["technique"]
            mdl = v3r["model"]
            label = f"{tech}+{mdl}"

            v2_match = v2_res[
                (v2_res["coin"] == coin) &
                (v2_res["technique"] == tech) &
                (v2_res["model"] == mdl)
            ]
            if len(v2_match) > 0:
                v2_f2 = v2_match.iloc[0]["f2_mean"]
                v2_auc = v2_match.iloc[0]["auc_pr_mean"]
                v3_f2 = v3r["f2_mean"]
                v3_auc = v3r["auc_pr_mean"]
                diff = v3_f2 - v2_f2
                print(f"  {label:30s} | {v2_f2:8.3f} | {v3_f2:8.3f} | {diff:+8.3f} | {v2_auc:9.3f} | {v3_auc:9.3f}")

    # 저장
    v3_df.to_csv(os.path.join(ML_DIR, "v3_model_results.csv"), index=False)
    print(f"\n결과 저장: data/ml/v3_model_results.csv")


if __name__ == "__main__":
    main()
