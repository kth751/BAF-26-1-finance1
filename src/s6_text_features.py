"""
s6_text_features.py — 뉴스 텍스트 기반 파생변수 생성

1. 키워드/이벤트 빈도: 위험 키워드 사전 기반 일별 빈도
2. 토픽 모델링 (LDA): 뉴스 토픽 비율 일별 집계

입력: data/raw/news_raw.csv
출력: data/processed/text_features.csv
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RAW_PATH = os.path.join(PROJECT_DIR, "data", "raw", "news_raw.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "data", "processed")

# ═══════════════════════════════════════════════════════════════════
# 1. 키워드/이벤트 빈도
# ═══════════════════════════════════════════════════════════════════

RISK_KEYWORDS = {
    "depeg": ["depeg", "de-peg", "depegging", "lost peg", "broke peg", "off peg"],
    "bankrun": ["bank run", "bankrun", "withdraw", "withdrawal", "redeem", "redemption"],
    "liquidation": ["liquidat", "margin call", "forced sell", "collateral call"],
    "hack": ["hack", "exploit", "vulnerability", "breach", "attack", "compromised"],
    "insolvency": ["insolvent", "insolvency", "bankrupt", "default", "collapse"],
    "regulation": ["regulat", "sec ", "cftc", "ban", "crackdown", "lawsuit", "subpoena"],
    "panic": ["panic", "crash", "dump", "plunge", "freefall", "capitulat"],
    "contagion": ["contagion", "spillover", "systemic", "domino", "cascad"],
}

POSITIVE_KEYWORDS = {
    "stability": ["stable", "stability", "resilient", "recover", "peg restored"],
    "adoption": ["adopt", "integrat", "partner", "launch", "listing"],
    "backing": ["reserve", "audit", "attestation", "backed", "collateral ratio"],
}


def compute_keyword_features(df):
    """기사별 키워드 매칭 → 일별 집계"""
    # title + text 합치기
    df["full_text"] = (
        df["title"].fillna("") + " " + df["text"].fillna("")
    ).str.lower()

    # 리스크 키워드 매칭
    for category, keywords in RISK_KEYWORDS.items():
        pattern = "|".join(re.escape(k) for k in keywords)
        df[f"kw_{category}"] = df["full_text"].str.contains(pattern, regex=True).astype(int)

    # 긍정 키워드 매칭
    for category, keywords in POSITIVE_KEYWORDS.items():
        pattern = "|".join(re.escape(k) for k in keywords)
        df[f"kw_{category}"] = df["full_text"].str.contains(pattern, regex=True).astype(int)

    # 일별 집계
    risk_cols = [f"kw_{c}" for c in RISK_KEYWORDS.keys()]
    pos_cols = [f"kw_{c}" for c in POSITIVE_KEYWORDS.keys()]
    all_kw_cols = risk_cols + pos_cols

    daily = df.groupby("Date")[all_kw_cols].sum().reset_index()

    # 일별 기사 수
    daily_count = df.groupby("Date").size().reset_index(name="news_total")
    daily = daily.merge(daily_count, on="Date")

    # 비율 변수
    for col in risk_cols:
        daily[f"{col}_ratio"] = daily[col] / daily["news_total"].replace(0, np.nan)

    # 종합 리스크 점수
    daily["risk_keyword_total"] = daily[risk_cols].sum(axis=1)
    daily["risk_keyword_ratio"] = daily["risk_keyword_total"] / daily["news_total"].replace(0, np.nan)
    daily["positive_keyword_total"] = daily[pos_cols].sum(axis=1)
    daily["risk_positive_gap"] = daily["risk_keyword_total"] - daily["positive_keyword_total"]

    # lag 변수
    for lag in [1, 3, 7]:
        daily[f"risk_keyword_total_lag{lag}"] = daily["risk_keyword_total"].shift(lag)
        daily[f"risk_keyword_ratio_lag{lag}"] = daily["risk_keyword_ratio"].shift(lag)

    # rolling 평균
    daily["risk_keyword_ma7"] = daily["risk_keyword_total"].rolling(7).mean()
    daily["risk_keyword_spike"] = (
        daily["risk_keyword_total"] > daily["risk_keyword_ma7"] * 2
    ).astype(int)

    print(f"키워드 변수: {len([c for c in daily.columns if c != 'Date'])}개")
    print(f"  리스크 키워드 카테고리: {list(RISK_KEYWORDS.keys())}")
    print(f"  긍정 키워드 카테고리: {list(POSITIVE_KEYWORDS.keys())}")

    return daily


# ═══════════════════════════════════════════════════════════════════
# 2. 토픽 모델링 (LDA)
# ═══════════════════════════════════════════════════════════════════

N_TOPICS = 8


def compute_topic_features(df):
    """LDA 토픽 모델링 → 일별 토픽 비율"""
    # 텍스트 준비
    texts = (df["title"].fillna("") + " " + df["text"].fillna("")).tolist()

    # TF-IDF
    print(f"\nTF-IDF 벡터화 중... ({len(texts)}건)")
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        min_df=5,
        max_df=0.95,
    )
    tfidf_matrix = tfidf.fit_transform(texts)
    print(f"  TF-IDF: {tfidf_matrix.shape}")

    # LDA (CountVectorizer 기반이 더 적합)
    print(f"LDA ({N_TOPICS} topics) 학습 중...")
    count_vec = CountVectorizer(
        max_features=5000,
        stop_words="english",
        min_df=5,
        max_df=0.95,
    )
    count_matrix = count_vec.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=42,
        max_iter=20,
        learning_method="online",
        n_jobs=-1,
    )
    topic_dist = lda.fit_transform(count_matrix)
    print(f"  LDA 완료: {topic_dist.shape}")

    # 토픽별 상위 키워드 출력
    feature_names = count_vec.get_feature_names_out()
    print(f"\n  토픽별 상위 키워드:")
    for i, topic in enumerate(lda.components_):
        top_words = [feature_names[j] for j in topic.argsort()[-10:][::-1]]
        print(f"    Topic {i}: {', '.join(top_words)}")

    # 기사별 토픽 비율 → DataFrame
    topic_cols = [f"topic_{i}" for i in range(N_TOPICS)]
    df_topics = pd.DataFrame(topic_dist, columns=topic_cols)
    df_topics["Date"] = df["Date"].values

    # 일별 평균 토픽 비율
    daily_topics = df_topics.groupby("Date")[topic_cols].mean().reset_index()

    # 일별 지배적 토픽
    daily_topics["dominant_topic"] = daily_topics[topic_cols].idxmax(axis=1)
    daily_topics["topic_concentration"] = daily_topics[topic_cols].max(axis=1)

    # 토픽 엔트로피 (다양성 지표)
    topic_vals = daily_topics[topic_cols].values
    topic_vals = np.clip(topic_vals, 1e-10, 1)
    daily_topics["topic_entropy"] = -np.sum(topic_vals * np.log(topic_vals), axis=1)

    # lag 변수 (주요 토픽)
    for lag in [1, 3]:
        daily_topics[f"topic_entropy_lag{lag}"] = daily_topics["topic_entropy"].shift(lag)
        daily_topics[f"topic_concentration_lag{lag}"] = daily_topics["topic_concentration"].shift(lag)

    print(f"\n토픽 변수: {len([c for c in daily_topics.columns if c != 'Date'])}개")

    return daily_topics


# ═══════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("뉴스 텍스트 파생변수 생성")
    print("=" * 60)

    df = pd.read_csv(RAW_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"원본: {df.shape[0]}건, {df['Date'].min().date()} ~ {df['Date'].max().date()}")

    # 1. 키워드 빈도
    print("\n" + "-" * 40)
    print("1. 키워드/이벤트 빈도")
    print("-" * 40)
    kw_daily = compute_keyword_features(df)

    # 2. 토픽 모델링
    print("\n" + "-" * 40)
    print("2. 토픽 모델링 (LDA)")
    print("-" * 40)
    topic_daily = compute_topic_features(df)

    # 병합
    result = kw_daily.merge(topic_daily, on="Date", how="outer")
    result = result.sort_values("Date").reset_index(drop=True)

    # 저장
    out_path = os.path.join(OUT_DIR, "text_features.csv")
    result.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print(f"저장 완료: {out_path}")
    print(f"  기간: {result['Date'].min().date()} ~ {result['Date'].max().date()}")
    print(f"  행: {result.shape[0]}, 변수: {result.shape[1] - 1}개 (Date 제외)")
    print("=" * 60)

    # 디페깅 시기 키워드 확인
    print("\n디페깅 주요 시기 키워드 분포:")
    for period, start, end in [
        ("코로나 (2020-03)", "2020-03-01", "2020-03-31"),
        ("DeFi Summer (2020-08)", "2020-08-01", "2020-08-31"),
        ("SVB (2023-03)", "2023-03-01", "2023-03-31"),
    ]:
        sub = result[(result["Date"] >= start) & (result["Date"] <= end)]
        if len(sub) > 0:
            print(f"  {period}: risk_total={sub['risk_keyword_total'].sum():.0f}, "
                  f"depeg_kw={sub['kw_depeg'].sum():.0f}, "
                  f"panic_kw={sub['kw_panic'].sum():.0f}, "
                  f"bankrun_kw={sub['kw_bankrun'].sum():.0f}")


if __name__ == "__main__":
    main()
