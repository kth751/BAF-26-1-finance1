"""
s7_feature_count_experiment.py — 변수 개수별 성능 비교

텍스트 포함 전체 풀에서 Step1(MI) + Step2(다중공선성) 후:
  A. Top 60 (기존)
  B. Top 80
  C. 전체 (Step2까지만, Top N 없음)

각 설정별로 USDC(ADASYN+XGB), DAI(SMOTE_05+RF) CV 성능 비교.
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
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN, SMOTE

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PROC_DIR = os.path.join(PROJECT_DIR, "data", "processed")
ML_DIR = os.path.join(PROJECT_DIR, "data", "ml")

RANDOM_STATE = 42
N_SPLITS = 5

EXCLUDE_PATTERNS = [
    "depeg_", "depeg_raw_", "tp_",
    "V_monthly_", "thresh_low_", "thresh_high_",
]

LOG_PATTERNS = [
    "market_cap_", "circ_", "volume_", "vol_ma7_",
    "m2_supply", "fed_balance_sheet",
    "eth_circ_", "eth_minted_", "unreleased_",
    "tvl_", "lending_tvl_total", "lending_tvl_total_ma7",
    "eth_daily_fees_usd", "eth_fees_ma7",
]


# ── 데이터 준비 ──

def load_text_features():
    tf = pd.read_csv(os.path.join(PROC_DIR, "text_features.csv"), parse_dates=["Date"])
    emb = pd.read_csv(os.path.join(PROC_DIR, "embedding_features.csv"), parse_dates=["Date"])
    if "dominant_topic" in tf.columns:
        tf = tf.drop(columns=["dominant_topic"])
    return tf.merge(emb, on="Date", how="outer")


def load_coin_data(coin, text_df):
    df = pd.read_csv(os.path.join(PROC_DIR, f"df_{coin.lower()}.csv"), parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.merge(text_df, on="Date", how="left")

    df["Y"] = df[f"depeg_{coin}"].shift(-1)
    df = df.dropna(subset=["Y"]).copy()
    df["Y"] = df["Y"].astype(int)

    exclude = ["Date", "Y", "fgi_class"]
    for col in df.columns:
        for pat in EXCLUDE_PATTERNS:
            if col.startswith(pat) or col == f"depeg_{coin}":
                exclude.append(col)
                break
    exclude = list(set(exclude))
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]

    X = df[feature_cols].copy()
    Y = df["Y"].copy()
    return X, Y


def step1_mi_filter(X, Y):
    X = X.replace([np.inf, -np.inf], np.nan)
    zero_var = X.columns[X.var() == 0].tolist()
    X = X.drop(columns=zero_var)
    high_miss = X.columns[X.isnull().mean() >= 0.5].tolist()
    X = X.drop(columns=high_miss)

    imp = SimpleImputer(strategy="median")
    X_filled = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    mi_scores = mutual_info_classif(X_filled, Y, random_state=RANDOM_STATE, n_neighbors=5)
    mi_series = pd.Series(mi_scores, index=X.columns)
    low_mi = mi_series[mi_series < 0.001].index.tolist()
    X = X.drop(columns=low_mi)
    return X


def step1b_log_transform(X):
    for col in X.columns:
        if any(col.startswith(pat) or col == pat for pat in LOG_PATTERNS):
            if X[col].min() >= 0 and X[col].max() > 1000:
                X[col] = np.log1p(X[col])
    return X


def step2_collinearity_filter(X, Y, threshold=0.95):
    imp = SimpleImputer(strategy="median")
    X_filled = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    corr_matrix = X_filled.corr().abs()
    corr_with_y = X_filled.corrwith(Y).abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        for hc in upper.index[upper[col] >= threshold].tolist():
            if hc in to_drop or col in to_drop:
                continue
            if corr_with_y.get(col, 0) >= corr_with_y.get(hc, 0):
                to_drop.add(hc)
            else:
                to_drop.add(col)
    X = X.drop(columns=list(to_drop))
    return X


def step3_rf_importance(X, Y, top_n):
    imp = SimpleImputer(strategy="median")
    X_filled = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_filled, Y)
    importance = pd.Series(rf.feature_importances_, index=X.columns)
    selected = importance.nlargest(top_n).index.tolist()
    return X[selected], selected


# ── CV 평가 ──

def fbeta(y_true, y_pred, beta=2.0):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    if p + r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r)


def run_cv(X, Y, coin, skf):
    """코인별 최적 기법으로 CV 실행"""
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        n_pos = int(Y_train.sum())
        n_neg = int((Y_train == 0).sum())
        if n_pos < 2:
            continue

        imp = SimpleImputer(strategy="median")
        X_train = imp.fit_transform(X_train)
        X_test = imp.transform(X_test)

        # 코인별 최적 기법
        if coin == "USDC":
            k = min(5, n_pos - 1)
            if k < 1:
                continue
            try:
                sampler = ADASYN(sampling_strategy=0.5, n_neighbors=k, random_state=RANDOM_STATE)
                X_train, Y_train = sampler.fit_resample(X_train, Y_train)
            except Exception:
                continue
            n_pos_r = int((Y_train == 1).sum())
            n_neg_r = int((Y_train == 0).sum())
            model = XGBClassifier(
                n_estimators=300, max_depth=6,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=1, eval_metric="logloss",
                random_state=RANDOM_STATE, verbosity=0, n_jobs=1
            )
        else:  # DAI
            k = min(5, n_pos - 1)
            if k < 1:
                continue
            try:
                sampler = SMOTE(sampling_strategy=0.5, k_neighbors=k, random_state=RANDOM_STATE)
                X_train, Y_train = sampler.fit_resample(X_train, Y_train)
            except Exception:
                continue
            model = RandomForestClassifier(
                n_estimators=300, max_depth=8,
                random_state=RANDOM_STATE, n_jobs=1
            )

        model.fit(X_train, Y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        if len(np.unique(Y_test)) < 2:
            continue

        fold_metrics.append({
            "auc_roc": roc_auc_score(Y_test, y_prob),
            "auc_pr": average_precision_score(Y_test, y_prob),
            "recall": recall_score(Y_test, y_pred, zero_division=0),
            "precision": precision_score(Y_test, y_pred, zero_division=0),
            "f1": f1_score(Y_test, y_pred, zero_division=0),
            "f2": fbeta(Y_test, y_pred),
        })

    if not fold_metrics:
        return None

    df_f = pd.DataFrame(fold_metrics)
    result = {}
    for m in ["auc_roc", "auc_pr", "recall", "precision", "f1", "f2"]:
        result[f"{m}_mean"] = df_f[m].mean()
        result[f"{m}_std"] = df_f[m].std()
    return result


# ── 메인 ──

def main():
    text_df = load_text_features()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    configs = [
        {"name": "Top 60", "top_n": 60},
        {"name": "Top 80", "top_n": 80},
        {"name": "전체 (Step2까지)", "top_n": None},
    ]

    all_results = []

    for coin in ["USDC", "DAI"]:
        technique = "ADASYN+XGB" if coin == "USDC" else "SMOTE_05+RF"

        print(f"\n{'='*65}")
        print(f"  {coin} 변수 개수별 성능 비교 ({technique})")
        print(f"{'='*65}")

        X, Y = load_coin_data(coin, text_df)
        print(f"  전체 후보: {X.shape[1]}개")

        X = step1_mi_filter(X, Y)
        print(f"  Step1 (MI 필터) 후: {X.shape[1]}개")

        X = step1b_log_transform(X)

        X_s2 = step2_collinearity_filter(X, Y)
        print(f"  Step2 (다중공선성) 후: {X_s2.shape[1]}개")

        for cfg in configs:
            top_n = cfg["top_n"]
            label = cfg["name"]

            if top_n is not None:
                X_final, selected = step3_rf_importance(X_s2, Y, top_n)
            else:
                X_final = X_s2
                selected = X_s2.columns.tolist()

            n_features = X_final.shape[1]
            t0 = time.time()
            result = run_cv(X_final.values, Y.values, coin, skf)
            elapsed = time.time() - t0

            if result is None:
                print(f"  {label:20s} ({n_features:3d}개) → FAILED")
                continue

            # 텍스트 변수 개수
            text_prefixes = ["kw_", "risk_", "positive_", "topic_", "emb_"]
            n_text = len([c for c in selected if any(c.startswith(p) for p in text_prefixes)])

            print(f"  {label:20s} ({n_features:3d}개, 텍스트={n_text:2d}) → "
                  f"F2={result['f2_mean']:.3f}±{result['f2_std']:.3f}  "
                  f"Recall={result['recall_mean']:.3f}  "
                  f"Prec={result['precision_mean']:.3f}  "
                  f"AUC-PR={result['auc_pr_mean']:.3f}  ({elapsed:.1f}s)")

            all_results.append({
                "coin": coin, "technique": technique,
                "config": label, "n_features": n_features, "n_text": n_text,
                **result
            })

    # v2 (텍스트 없음) 결과도 비교용으로 추가
    print(f"\n{'='*65}")
    print(f"  v2 (텍스트 없음) 비교 기준")
    print(f"{'='*65}")

    for coin in ["USDC", "DAI"]:
        technique = "ADASYN+XGB" if coin == "USDC" else "SMOTE_05+RF"
        v2_X = pd.read_csv(os.path.join(ML_DIR, f"v2_X_{coin.lower()}.csv"))
        v2_X = v2_X.replace([np.inf, -np.inf], np.nan)
        v2_Y = pd.read_csv(os.path.join(ML_DIR, f"v2_Y_{coin.lower()}.csv"), header=None).squeeze().values

        t0 = time.time()
        result = run_cv(v2_X.values, v2_Y, coin, skf)
        elapsed = time.time() - t0

        if result:
            print(f"  {coin:6s} v2 (60개, 텍스트=0)  → "
                  f"F2={result['f2_mean']:.3f}±{result['f2_std']:.3f}  "
                  f"Recall={result['recall_mean']:.3f}  "
                  f"Prec={result['precision_mean']:.3f}  "
                  f"AUC-PR={result['auc_pr_mean']:.3f}  ({elapsed:.1f}s)")

            all_results.append({
                "coin": coin, "technique": technique,
                "config": "v2 (텍스트 없음)", "n_features": 60, "n_text": 0,
                **result
            })

    # 결과 저장
    df = pd.DataFrame(all_results)
    out_path = os.path.join(ML_DIR, "feature_count_experiment.csv")
    df.to_csv(out_path, index=False)

    # 최종 요약
    print(f"\n{'='*65}")
    print(f"  최종 비교 요약")
    print(f"{'='*65}")
    for coin in ["USDC", "DAI"]:
        print(f"\n  --- {coin} ---")
        sub = df[df["coin"] == coin].sort_values("f2_mean", ascending=False)
        for _, r in sub.iterrows():
            print(f"  {r['config']:20s} ({int(r['n_features']):3d}개) → F2={r['f2_mean']:.4f}±{r['f2_std']:.4f}  "
                  f"AUC-PR={r['auc_pr_mean']:.4f}")

    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
