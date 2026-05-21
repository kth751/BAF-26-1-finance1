"""
s1_prepare_data.py — Feature Selection + 해석성 전처리 + ML 데이터 생성

파이프라인:
  Step 1: 기계적 필터 (분산=0, 결측50%+, MI<0.001)
  Step 1b: Winsorize (상하위 1% clip) — 이상치가 SHAP을 왜곡하는 것 방지
  Step 1c: Log 변환 (고왜도 수준값) — SHAP dependence plot 해석성 개선
  Step 2: 다중공선성 제거 (r≥0.95)
  Step 3: RF Importance Top 60

입력: data/processed/df_usdc.csv, df_dai.csv
출력: data/ml/v2_X_{coin}.csv, v2_Y_{coin}.csv, v2_selected_features_{coin}.csv,
      v2_feature_categories_{coin}.csv
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PROC_DIR = os.path.join(PROJECT_DIR, "data", "processed")
ML_DIR = os.path.join(PROJECT_DIR, "data", "ml")
os.makedirs(ML_DIR, exist_ok=True)

RANDOM_STATE = 42
TOP_N = 60

EXCLUDE_PATTERNS = [
    "depeg_", "depeg_raw_", "tp_",
    "V_monthly_", "thresh_low_", "thresh_high_",
]

# log 변환 대상: 양수이고 스케일이 큰 수준값 변수 패턴
LOG_PATTERNS = [
    "market_cap_", "circ_", "volume_", "vol_ma7_",
    "m2_supply", "fed_balance_sheet",
    "eth_circ_", "eth_minted_", "unreleased_",
    "tvl_", "lending_tvl_total", "lending_tvl_total_ma7",
    "eth_daily_fees_usd", "eth_fees_ma7",
]

# 카테고리 분류 규칙
CATEGORY_RULES = [
    ("macro", ["dxy", "vix", "federal_funds_rate", "rate_hike", "risk_off",
               "us_2y_", "us_3m_", "us_5y_", "us_10y_",
               "yield_spread", "yield_curve",
               "credit_spread", "credit_stress",
               "m2_", "fed_b", "qt_signal", "liquidity_stress"]),
    ("crypto_market", ["btc_", "eth_return", "eth_vol", "crypto_stress",
                       "btc_dominance", "btc_dom_change"]),
    ("sentiment", ["fgi", "extreme_fear", "extreme_greed",
                   "gt_", "news_", "emb_", "kw_", "risk_keyword",
                   "positive_keyword", "risk_positive"]),
    ("onchain", ["supply_change", "supply_growth", "mint_intensity",
                 "eth_circ_", "eth_minted_", "unreleased_",
                 "gas_price", "eth_fees", "eth_daily_fees"]),
    ("defi", ["tvl_", "lending_tvl"]),
    ("cross_coin", ["usdt_usdc_", "usdt_dai_", "vol_share_",
                    "dai_eth_return_corr"]),
]


def load_coin_data(coin):
    path = os.path.join(PROC_DIR, f"df_{coin.lower()}.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

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

    print(f"[{coin}] 로드 완료: {X.shape[0]}행, {X.shape[1]}변수")
    print(f"  디페깅 비율: {Y.mean()*100:.1f}% ({Y.sum()}건)")
    return X, Y


def step1_mechanical_filter(X, Y):
    n_before = X.shape[1]

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

    print(f"  Step1: {n_before} → {X.shape[1]}개 (분산0={len(zero_var)}, 결측50%+={len(high_miss)}, MI<0.001={len(low_mi)})")
    return X


def step1b_winsorize(X, lower=0.01, upper=0.99):
    """Winsorize 제거됨 — 디페깅 극단 이벤트에서 극단값 자체가 핵심 신호이므로 clip하지 않음"""
    print(f"  Step1b: Winsorize 미적용 (극단값 = 디페깅 신호, 보존)")
    return X


def step1c_log_transform(X):
    """고왜도 수준값 변수에 log1p 변환 (원본 교체)"""
    transformed = []
    for col in X.columns:
        is_log_target = any(col.startswith(pat) or col == pat for pat in LOG_PATTERNS)
        if not is_log_target:
            continue
        # 양수인 경우만 변환
        if X[col].min() >= 0 and X[col].max() > 1000:
            X[col] = np.log1p(X[col])
            transformed.append(col)

    print(f"  Step1c: Log 변환 ({len(transformed)}개): {transformed[:5]}{'...' if len(transformed) > 5 else ''}")
    return X


def step2_collinearity_filter(X, Y, threshold=0.95):
    n_before = X.shape[1]
    imp = SimpleImputer(strategy="median")
    X_filled = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    corr_matrix = X_filled.corr().abs()
    corr_with_y = X_filled.corrwith(Y).abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()

    for col in upper.columns:
        high_corr_cols = upper.index[upper[col] >= threshold].tolist()
        for hc in high_corr_cols:
            if hc in to_drop or col in to_drop:
                continue
            if corr_with_y.get(col, 0) >= corr_with_y.get(hc, 0):
                to_drop.add(hc)
            else:
                to_drop.add(col)

    X = X.drop(columns=list(to_drop))
    print(f"  Step2: {n_before} → {X.shape[1]}개 (다중공선성 제거={len(to_drop)})")
    return X


def step3_rf_importance(X, Y, top_n=TOP_N):
    n_before = X.shape[1]
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

    X = X[selected]
    print(f"  Step3: {n_before} → {X.shape[1]}개 (RF Importance Top {top_n})")
    return X, selected


def categorize_features(features):
    """변수 목록에 카테고리 태깅"""
    result = []
    for feat in features:
        category = "price_volatility"  # 기본값 (코인 자체 가격/변동성)
        for cat_name, patterns in CATEGORY_RULES:
            if any(feat.startswith(p) or feat == p for p in patterns):
                category = cat_name
                break
        result.append({"feature": feat, "category": category})
    return pd.DataFrame(result)


def main():
    for coin in ["USDC", "DAI"]:
        print(f"\n{'='*60}")
        print(f"  {coin} Feature Selection")
        print(f"{'='*60}")

        X, Y = load_coin_data(coin)
        X = step1_mechanical_filter(X, Y)
        X = step1b_winsorize(X)
        X = step1c_log_transform(X)
        X = step2_collinearity_filter(X, Y)
        X, selected = step3_rf_importance(X, Y)

        # 카테고리 태깅
        cat_df = categorize_features(selected)
        cat_summary = cat_df["category"].value_counts()

        # 저장
        X.to_csv(os.path.join(ML_DIR, f"v2_X_{coin.lower()}.csv"), index=False)
        Y.to_csv(os.path.join(ML_DIR, f"v2_Y_{coin.lower()}.csv"), index=False, header=False)
        pd.DataFrame({"feature": selected}).to_csv(
            os.path.join(ML_DIR, f"v2_selected_features_{coin.lower()}.csv"), index=False
        )
        cat_df.to_csv(
            os.path.join(ML_DIR, f"v2_feature_categories_{coin.lower()}.csv"), index=False
        )

        print(f"\n  저장 완료: v2_X_{coin.lower()}.csv ({X.shape})")
        print(f"  카테고리 분포:")
        for cat, cnt in cat_summary.items():
            print(f"    {cat:20s}: {cnt}개")


if __name__ == "__main__":
    main()
