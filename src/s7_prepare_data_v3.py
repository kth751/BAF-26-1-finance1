"""
s7_prepare_data_v3.py — v2 변수 고정 + 텍스트 변수 추가

전략: v2에서 선택된 60개 변수를 그대로 유지하고,
텍스트 변수(키워드+토픽+임베딩) 중 유의미한 것만 추가 (60 + α).
→ v2 성능을 바닥으로 보장하면서 텍스트 기여도를 순수 평가.

텍스트 변수 선별: MI > 0.001 + RF Importance 기준 상위 선택
v2 변수와 다중공선성(r≥0.95) 있는 텍스트 변수는 제외.

입력: data/ml/v2_X_{coin}.csv, v2_Y_{coin}.csv, v2_selected_features_{coin}.csv,
      data/processed/text_features.csv, embedding_features.csv, df_{coin}.csv
출력: data/ml/v3_X_{coin}.csv, v3_Y_{coin}.csv, v3_selected_features_{coin}.csv
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
TEXT_PREFIXES = ["kw_", "risk_", "positive_", "topic_", "emb_"]


def load_text_features():
    tf_path = os.path.join(PROC_DIR, "text_features.csv")
    emb_path = os.path.join(PROC_DIR, "embedding_features.csv")

    tf = pd.read_csv(tf_path, parse_dates=["Date"])
    emb = pd.read_csv(emb_path, parse_dates=["Date"])

    if "dominant_topic" in tf.columns:
        tf = tf.drop(columns=["dominant_topic"])

    merged = tf.merge(emb, on="Date", how="outer")
    print(f"텍스트 변수 로드: {merged.shape[1] - 1}개 (키워드+토픽: {tf.shape[1]-1}, 임베딩: {emb.shape[1]-1})")
    return merged


def load_v2_and_text(coin, text_df):
    # v2 데이터 로드
    v2_feats = pd.read_csv(os.path.join(ML_DIR, f"v2_selected_features_{coin.lower()}.csv"))["feature"].tolist()
    v2_Y = pd.read_csv(os.path.join(ML_DIR, f"v2_Y_{coin.lower()}.csv"), header=None).squeeze().values

    # 원본 데이터에서 텍스트 변수 포함한 전체 로드
    df = pd.read_csv(os.path.join(PROC_DIR, f"df_{coin.lower()}.csv"), parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.merge(text_df, on="Date", how="left")

    df["Y"] = df[f"depeg_{coin}"].shift(-1)
    df = df.dropna(subset=["Y"]).copy()
    df["Y"] = df["Y"].astype(int)

    # 텍스트 변수만 추출
    text_cols = [c for c in df.columns
                 if any(c.startswith(p) for p in TEXT_PREFIXES)
                 and c not in v2_feats
                 and df[c].dtype in [np.float64, np.int64, float, int]]

    X_text = df[text_cols].copy()
    X_text = X_text.replace([np.inf, -np.inf], np.nan)

    print(f"[{coin}] v2 변수: {len(v2_feats)}개 (고정)")
    print(f"  텍스트 후보: {len(text_cols)}개")
    print(f"  디페깅 비율: {df['Y'].mean()*100:.1f}% ({df['Y'].sum()}건)")

    return v2_feats, X_text, df["Y"].values, df


def filter_text_variables(X_text, Y, v2_X_df):
    """텍스트 변수 필터링: MI > 0.001 + v2와 다중공선성 없는 것만"""
    n_before = X_text.shape[1]

    # 1. 분산 0, 결측 50%+ 제거
    zero_var = X_text.columns[X_text.var() == 0].tolist()
    X_text = X_text.drop(columns=zero_var)

    high_miss = X_text.columns[X_text.isnull().mean() >= 0.5].tolist()
    X_text = X_text.drop(columns=high_miss)

    if X_text.shape[1] == 0:
        print(f"  텍스트 필터: {n_before} → 0개 (전부 제거)")
        return X_text

    # 2. MI > 0.001
    imp = SimpleImputer(strategy="median")
    X_filled = pd.DataFrame(imp.fit_transform(X_text), columns=X_text.columns)
    mi_scores = mutual_info_classif(X_filled, Y, random_state=RANDOM_STATE, n_neighbors=5)
    mi_series = pd.Series(mi_scores, index=X_text.columns)
    low_mi = mi_series[mi_series < 0.001].index.tolist()
    X_text = X_text.drop(columns=low_mi)

    if X_text.shape[1] == 0:
        print(f"  텍스트 필터: {n_before} → 0개 (MI 필터 후 전부 제거)")
        return X_text

    # 3. v2 변수와 다중공선성(r≥0.95) 체크
    imp2 = SimpleImputer(strategy="median")
    X_text_filled = pd.DataFrame(imp2.fit_transform(X_text), columns=X_text.columns)
    v2_filled = v2_X_df.copy()

    collinear = []
    for tc in X_text_filled.columns:
        for vc in v2_filled.columns:
            r = np.corrcoef(X_text_filled[tc].values, v2_filled[vc].values)[0, 1]
            if abs(r) >= 0.95:
                collinear.append(tc)
                break
    X_text = X_text.drop(columns=collinear)

    print(f"  텍스트 필터: {n_before} → {X_text.shape[1]}개 "
          f"(분산0={len(zero_var)}, 결측={len(high_miss)}, MI<0.001={len(low_mi)}, 다중공선성={len(collinear)})")
    return X_text


def select_text_by_importance(X_text, Y, v2_X, v2_feats):
    """v2 + 텍스트 합쳐서 RF 학습 → 텍스트 변수 중 importance 상위만 선택"""
    if X_text.shape[1] == 0:
        return [], pd.Series(dtype=float)

    X_combined = np.hstack([v2_X, X_text.values])
    all_cols = v2_feats + X_text.columns.tolist()

    imp = SimpleImputer(strategy="median")
    X_filled = imp.fit_transform(X_combined)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_filled, Y)

    importance = pd.Series(rf.feature_importances_, index=all_cols)
    text_imp = importance[X_text.columns].sort_values(ascending=False)

    # v2 변수의 최소 importance를 기준선으로 사용
    v2_min_imp = importance[v2_feats].min()
    selected = text_imp[text_imp > v2_min_imp].index.tolist()

    print(f"  RF Importance 기준선 (v2 최소): {v2_min_imp:.4f}")
    print(f"  텍스트 변수 선택: {len(selected)}개 / {X_text.shape[1]}개")
    for c in selected:
        print(f"    - {c} (importance: {text_imp[c]:.4f})")

    return selected, text_imp


def main():
    text_df = load_text_features()

    for coin in ["USDC", "DAI"]:
        print(f"\n{'='*60}")
        print(f"  {coin} v3: v2 고정 + 텍스트 추가")
        print(f"{'='*60}")

        v2_feats, X_text, Y, df = load_v2_and_text(coin, text_df)

        # v2 X 로드 (imputed 상태로)
        v2_X = pd.read_csv(os.path.join(ML_DIR, f"v2_X_{coin.lower()}.csv"))
        v2_X = v2_X.replace([np.inf, -np.inf], np.nan)

        imp_v2 = SimpleImputer(strategy="median")
        v2_X_filled = pd.DataFrame(imp_v2.fit_transform(v2_X), columns=v2_X.columns)

        # 텍스트 변수 필터링
        X_text_filtered = filter_text_variables(X_text, Y, v2_X_filled)

        # RF importance로 텍스트 변수 최종 선택
        text_selected, text_imp = select_text_by_importance(
            X_text_filtered, Y, v2_X.values, v2_feats
        )

        # v3 = v2(60) + 텍스트(α)
        v3_feats = v2_feats + text_selected
        v3_X = df[v3_feats].copy()
        v3_X = v3_X.replace([np.inf, -np.inf], np.nan)

        print(f"\n  v3 최종: {len(v2_feats)} (v2) + {len(text_selected)} (텍스트) = {len(v3_feats)}개")

        # 저장
        v3_X.to_csv(os.path.join(ML_DIR, f"v3_X_{coin.lower()}.csv"), index=False)
        pd.Series(Y).to_csv(os.path.join(ML_DIR, f"v3_Y_{coin.lower()}.csv"), index=False, header=False)
        pd.DataFrame({"feature": v3_feats}).to_csv(
            os.path.join(ML_DIR, f"v3_selected_features_{coin.lower()}.csv"), index=False
        )
        print(f"  저장: v3_X_{coin.lower()}.csv ({v3_X.shape})")


if __name__ == "__main__":
    main()
