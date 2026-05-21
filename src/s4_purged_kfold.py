"""
s4_purged_kfold.py — Purged K-Fold vs StratifiedKFold 비교 실험

Purged K-Fold: 시간 순서 유지 + embargo(7일)로 fold 경계 데이터 제거
→ data leakage 영향 정량화
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN, SMOTE

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ML_DIR = os.path.join(PROJECT_DIR, "data", "ml")

RANDOM_STATE = 42
N_SPLITS = 5
EMBARGO = 7


def load_data(coin):
    X = pd.read_csv(os.path.join(ML_DIR, f"v2_X_{coin.lower()}.csv"))
    Y = pd.read_csv(os.path.join(ML_DIR, f"v2_Y_{coin.lower()}.csv"), header=None).squeeze()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.values, Y.values


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


def purged_kfold_split(n_samples, n_splits=5, embargo=7):
    """시간 순서 유지 + embargo로 fold 경계 데이터 제거"""
    indices = np.arange(n_samples)
    fold_size = n_samples // n_splits

    for i in range(n_splits):
        test_start = i * fold_size
        test_end = test_start + fold_size if i < n_splits - 1 else n_samples

        test_idx = indices[test_start:test_end]

        # embargo: test 전후 embargo일 제거
        purge_start = max(0, test_start - embargo)
        purge_end = min(n_samples, test_end + embargo)

        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[purge_start:purge_end] = False
        train_idx = indices[train_mask]

        yield train_idx, test_idx


def build_model(model_type, weight=False, n_pos=1, n_neg=1):
    if model_type == "RF":
        return RandomForestClassifier(
            n_estimators=300, max_depth=8,
            class_weight="balanced" if weight else None,
            random_state=RANDOM_STATE, n_jobs=1
        )
    else:
        spw = (n_neg / n_pos) if weight and n_pos > 0 else 1
        return XGBClassifier(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0, n_jobs=1
        )


def run_purged_experiment(X, Y, config):
    fold_metrics = []
    skipped = 0

    for fold_idx, (train_idx, test_idx) in enumerate(purged_kfold_split(len(Y), N_SPLITS, EMBARGO)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        n_pos = int(Y_train.sum())
        n_neg = int((Y_train == 0).sum())

        if n_pos < 2:
            skipped += 1
            continue

        # Impute
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

        model = build_model(config["model"], config["weight"], n_pos_r, n_neg_r)
        model.fit(X_train, Y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = evaluate(Y_test, y_prob)
        metrics["fold"] = fold_idx
        metrics["n_pos_test"] = int(Y_test.sum())
        fold_metrics.append(metrics)

    if not fold_metrics:
        return None, skipped

    df_f = pd.DataFrame(fold_metrics)
    result = {}
    for m in ["auc_roc", "auc_pr", "recall", "precision", "f1", "f2"]:
        vals = df_f[m].dropna()
        result[f"{m}_mean"] = vals.mean()
        result[f"{m}_std"] = vals.std()
    result["n_valid_folds"] = len(fold_metrics)
    return result, skipped


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

    all_results = []

    for coin in ["USDC", "DAI"]:
        X, Y = load_data(coin)
        n = len(Y)

        print("=" * 65)
        print(f"  {coin} - Purged K-Fold (n_splits={N_SPLITS}, embargo={EMBARGO})")
        print(f"  Data: {X.shape}, depeg: {int(Y.sum())} ({Y.mean()*100:.1f}%)")
        print("=" * 65)

        # fold info
        for fi, (tr, te) in enumerate(purged_kfold_split(n, N_SPLITS, EMBARGO)):
            purged = n - len(tr) - len(te)
            print(f"  fold {fi}: train={len(tr)} (pos={int(Y[tr].sum())}), "
                  f"test={len(te)} (pos={int(Y[te].sum())}), purged={purged}")
        print()

        for cfg in configs:
            label = f"{cfg['name']}+{cfg['model']}"
            result, skipped = run_purged_experiment(X, Y, cfg)

            if result is None:
                print(f"  {label:30s} -> FAILED (all folds skipped)")
                continue

            result["coin"] = coin
            result["technique"] = cfg["name"]
            result["model"] = cfg["model"]
            all_results.append(result)

            skip_msg = f"  ({skipped} fold skipped)" if skipped else ""
            print(f"  {label:30s} -> F2={result['f2_mean']:.3f}+/-{result['f2_std']:.3f}  "
                  f"Recall={result['recall_mean']:.3f}  Prec={result['precision_mean']:.3f}  "
                  f"AUC-PR={result['auc_pr_mean']:.3f}{skip_msg}")

    # === 비교 ===
    print("\n" + "=" * 65)
    print("  StratifiedKFold vs Purged K-Fold 비교")
    print("=" * 65)

    skf_res = pd.read_csv(os.path.join(ML_DIR, "imbalance_results.csv"))
    purged_df = pd.DataFrame(all_results)

    for coin in ["USDC", "DAI"]:
        print(f"\n  --- {coin} ---")
        header = f"  {'기법':30s} | {'SKF F2':>10s} | {'Purged F2':>10s} | {'차이':>8s} | {'SKF AUC-PR':>10s} | {'Purged AUC-PR':>13s}"
        print(header)
        print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*13}")

        for _, pr in purged_df[purged_df["coin"] == coin].iterrows():
            tech = pr["technique"]
            mdl = pr["model"]
            label = f"{tech}+{mdl}"

            skf_match = skf_res[
                (skf_res["coin"] == coin) &
                (skf_res["technique"] == tech) &
                (skf_res["model"] == mdl)
            ]
            if len(skf_match) > 0:
                skf_f2 = skf_match.iloc[0]["f2_mean"]
                skf_auc = skf_match.iloc[0]["auc_pr_mean"]
                pur_f2 = pr["f2_mean"]
                pur_auc = pr["auc_pr_mean"]
                diff = pur_f2 - skf_f2
                print(f"  {label:30s} | {skf_f2:10.3f} | {pur_f2:10.3f} | {diff:+8.3f} | {skf_auc:10.3f} | {pur_auc:13.3f}")

    # 결과 저장
    purged_df.to_csv(os.path.join(ML_DIR, "purged_kfold_results.csv"), index=False)
    print(f"\n결과 저장: data/ml/purged_kfold_results.csv")


if __name__ == "__main__":
    main()
