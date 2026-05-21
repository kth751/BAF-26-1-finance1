"""
전처리 및 파생변수 생성
출력:
  - processed/df_fiat.csv   : USDT + USDC (법정화폐 담보형, long format)
  - processed/df_dai.csv    : DAI (암호화폐 담보형)
  - processed/df_merged.csv : 전체 병합 (wide format, 나중에 통합 모델용)
"""

import pandas as pd
import numpy as np
import os

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RAW_DIR  = os.path.join(PROJECT_DIR, "data", "raw")
OUT_DIR  = os.path.join(PROJECT_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

START_DATE = "2020-01-01"


# ── 1. 각 데이터 로딩 및 개별 전처리 ─────────────────────────────────────────

def load_cmc():
    """CMC: OHLCV + 시가총액 (공급량 컬럼은 100% 결측 → 제거)"""
    df = pd.read_csv(os.path.join(RAW_DIR, "cmc_market_info.csv"))
    df["Date"] = pd.to_datetime(df["Date"])

    # 100% 결측 컬럼 제거 (circulating_supply, total_supply, max_supply)
    drop_cols = [c for c in df.columns if df[c].isnull().mean() == 1.0]
    df = df.drop(columns=drop_cols)

    # 필요한 컬럼만 유지
    keep = ["Date"] + [c for c in df.columns if any(
        c.startswith(p) for p in ["open_", "high_", "low_", "close_", "volume_", "market_cap_"]
    )]
    df = df[keep].sort_values("Date").reset_index(drop=True)
    print(f"CMC: {df.shape} | 결측 없음")
    return df


def load_macro():
    """거시경제: DXY, VIX (주말 결측 → ffill), 기준금리 (월별 발표 → ffill)"""
    df = pd.read_csv(os.path.join(RAW_DIR, "macro_data.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 주말/공휴일 결측 → forward fill
    df["dxy"] = df["dxy"].fillna(method="ffill")
    df["vix"] = df["vix"].fillna(method="ffill")
    # 기준금리: 월별 발표 → ffill 후 나머지 0으로 (초기값)
    df["federal_funds_rate"] = df["federal_funds_rate"].fillna(method="ffill").fillna(0)

    missing = df.isnull().sum().sum()
    print(f"Macro: {df.shape} | 잔여 결측: {missing}")
    return df


def load_fgi():
    """Fear & Greed Index: timestamp → Date 변환, 역순 정렬"""
    df = pd.read_csv(os.path.join(RAW_DIR, "fear_and_greed_index.csv"))
    df["Date"] = pd.to_datetime(df["timestamp"])
    df = df[["Date", "value", "value_classification"]].copy()
    df.columns = ["Date", "fgi", "fgi_class"]
    df = df.sort_values("Date").reset_index(drop=True)

    missing = df.isnull().sum().sum()
    print(f"FGI: {df.shape} | 결측: {missing}")
    return df


def load_onchain():
    """온체인 공급량: 100% 결측 컬럼 제거, 첫 행 NaN(diff 결과) → 0 처리"""
    df = pd.read_csv(os.path.join(RAW_DIR, "onchain_supply.csv"))
    df["Date"] = pd.to_datetime(df["Date"])

    # 100% 결측 컬럼 제거 (minted_USDT, minted_USDC, unreleased_DAI, minted_DAI)
    drop_cols = [c for c in df.columns if df[c].isnull().mean() == 1.0]
    df = df.drop(columns=drop_cols)

    # supply_change 첫 행 NaN → 0
    change_cols = [c for c in df.columns if "supply_change" in c]
    df[change_cols] = df[change_cols].fillna(0)

    # unreleased_USDC: 49% 결측 → 직전값 채움 후 나머지 0
    df["unreleased_USDC"] = df["unreleased_USDC"].fillna(method="ffill").fillna(0)

    # unreleased_USDT: 28% 결측 → ffill
    df["unreleased_USDT"] = df["unreleased_USDT"].fillna(method="ffill").fillna(0)

    df = df.sort_values("Date").reset_index(drop=True)
    missing = df.isnull().sum()
    print(f"Onchain: {df.shape} | 결측:\n{missing[missing > 0]}")
    return df


def load_google_trends():
    """Google Trends 검색량 (심리 프록시): 결측 → 0 (검색량 없음)"""
    path = os.path.join(RAW_DIR, "google_trends.csv")
    if not os.path.exists(path):
        print("Google Trends: 파일 없음 → 건너뜀")
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    gt_cols = [c for c in df.columns if c.startswith("gt_")]
    df[gt_cols] = df[gt_cols].fillna(0)
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"Google Trends: {df.shape} | 키워드: {len(gt_cols)}개")
    return df


def load_gas_price():
    """ETH 가스 가격 + 수수료: 주말 결측 → ffill"""
    path = os.path.join(RAW_DIR, "gas_price.csv")
    if not os.path.exists(path):
        print("Gas Price: 파일 없음 → 건너뜀")
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    num_cols = [c for c in df.columns if c != "Date"]
    df[num_cols] = df[num_cols].ffill()
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"Gas Price: {df.shape}")
    return df


def load_macro_additional():
    """추가 거시경제: 국채 수익률, 신용 스프레드, M2, Fed 대차대조표"""
    path = os.path.join(RAW_DIR, "macro_additional.csv")
    if not os.path.exists(path):
        print("Macro Additional: 파일 없음 → 건너뜀")
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    num_cols = [c for c in df.columns if c != "Date"]
    df[num_cols] = df[num_cols].ffill()
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"Macro Additional: {df.shape} | 변수: {num_cols}")
    return df


def load_defi_protocols():
    """DeFi Lending 프로토콜 TVL: 결측 → ffill"""
    path = os.path.join(RAW_DIR, "defi_protocols_tvl.csv")
    if not os.path.exists(path):
        print("DeFi Protocols: 파일 없음 → 건너뜀")
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    num_cols = [c for c in df.columns if c != "Date"]
    df[num_cols] = df[num_cols].ffill()
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"DeFi Protocols: {df.shape}")
    return df


# ── 2. 전체 병합 ──────────────────────────────────────────────────────────────

def merge_all(cmc, macro, fgi, onchain, gt=None, gas=None, defi_tvl=None, macro_add=None):
    """Date 기준 outer join 후 START_DATE 이후 필터링"""
    df = cmc.copy()
    df = df.merge(macro,   on="Date", how="left")
    df = df.merge(fgi,     on="Date", how="left")
    df = df.merge(onchain, on="Date", how="left")

    # 신규 데이터 병합
    if gt is not None:
        df = df.merge(gt, on="Date", how="left")
        gt_cols = [c for c in df.columns if c.startswith("gt_")]
        df[gt_cols] = df[gt_cols].fillna(0).astype(int)
    if gas is not None:
        df = df.merge(gas, on="Date", how="left")
        gas_cols = [c for c in df.columns if "gas_price" in c or "eth_fees" in c or "eth_daily" in c]
        df[gas_cols] = df[gas_cols].ffill()
    if defi_tvl is not None:
        df = df.merge(defi_tvl, on="Date", how="left")
        tvl_cols = [c for c in df.columns if c.startswith("tvl_") or c.startswith("lending_")]
        df[tvl_cols] = df[tvl_cols].ffill()
    if macro_add is not None:
        df = df.merge(macro_add, on="Date", how="left")
        macro_add_cols = [c for c in macro_add.columns if c != "Date"]
        df[macro_add_cols] = df[macro_add_cols].ffill()

    df = df[df["Date"] >= START_DATE].sort_values("Date").reset_index(drop=True)

    # FGI: 최대 ±1일 오차 허용 → ffill
    df["fgi"] = df["fgi"].ffill().bfill()
    df["fgi_class"] = df["fgi_class"].ffill().bfill()

    print(f"\n병합 완료: {df.shape}")
    missing_pct = (df.isnull().mean() * 100).round(1)
    print("결측률 > 0:\n", missing_pct[missing_pct > 0].to_string())
    return df


# ── 3. 타깃 변수 생성 (TP + ±1% 고정 + 2-of-3) ──────────────────────────────

def make_target(df, coin):
    """
    확정된 디페깅 정의:
      1) TP_t = (High_t + Low_t + Close_t) / 3
      2) D_t = 1 if |TP_t - 1| > 0.01, else 0  (±1% 고정 임계값)
      3) Depeg_t = 1 if D_t + D_{t-1} + D_{t-2} >= 2  (2-of-3 persistence)
    상방/하방 구분 없이 둘 다 디페깅으로 처리.
    """
    high_col  = f"high_{coin}"
    low_col   = f"low_{coin}"
    close_col = f"close_{coin}"

    # Step 1: Typical Price
    df[f"tp_{coin}"] = (df[high_col] + df[low_col] + df[close_col]) / 3

    # Step 2: 단일일 이탈 판정 (±1% 고정)
    depeg_raw = ((df[f"tp_{coin}"] - 1).abs() > 0.01).astype(int)
    df[f"depeg_raw_{coin}"] = depeg_raw

    # Step 3: 2-of-3 persistence filter (t, t-1, t-2 → 과거 방향만)
    rolling_sum = depeg_raw.rolling(3, min_periods=1).sum()
    df[f"depeg_{coin}"] = (rolling_sum >= 2).astype(int)

    n_raw   = depeg_raw.sum()
    n_depeg = df[f"depeg_{coin}"].sum()
    total   = len(df)
    print(f"{coin} 디페깅: raw={n_raw}일 → 2-of-3={n_depeg}일 / {total}일 ({n_depeg/total*100:.1f}%)")
    return df


# ── 4. 파생변수 생성 ──────────────────────────────────────────────────────────

def make_features(df, coin):
    """코인별 파생변수 생성"""
    c = coin  # 짧은 alias
    p = f"close_{c}"
    v = f"volume_{c}"
    m = f"market_cap_{c}"

    # ── 가격 기반 ──
    df[f"return_1d_{c}"]      = df[p].pct_change(1)
    df[f"return_3d_{c}"]      = df[p].pct_change(3)
    df[f"return_7d_{c}"]      = df[p].pct_change(7)
    df[f"depeg_magnitude_{c}"]= (df[p] - 1.0).abs()
    df[f"high_low_spread_{c}"]= (df[f"high_{c}"] - df[f"low_{c}"]) / df[p]
    df[f"upper_shadow_{c}"]   = (df[f"high_{c}"] - df[p]) / df[p]
    df[f"lower_shadow_{c}"]   = (df[p] - df[f"low_{c}"]) / df[p]

    # 이동평균 이탈
    df[f"ma7_{c}"]            = df[p].rolling(7).mean()
    df[f"ma30_{c}"]           = df[p].rolling(30).mean()
    df[f"ma7_dev_{c}"]        = df[p] - df[f"ma7_{c}"]
    df[f"ma30_dev_{c}"]       = df[p] - df[f"ma30_{c}"]

    # 변동성
    df[f"vol_7d_{c}"]         = df[f"return_1d_{c}"].rolling(7).std()
    df[f"vol_30d_{c}"]        = df[f"return_1d_{c}"].rolling(30).std()

    # RSI (14일)
    delta = df[p].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df[f"rsi_14_{c}"]         = 100 - (100 / (1 + rs))

    # ── 거래량 기반 ──
    df[f"vol_ma7_{c}"]        = df[v].rolling(7).mean()
    df[f"volume_ratio_{c}"]   = df[v] / df[f"vol_ma7_{c}"].replace(0, np.nan)
    df[f"volume_surge_{c}"]   = (df[f"volume_ratio_{c}"] > 2).astype(int)
    df[f"turnover_rate_{c}"]  = df[v] / df[m].replace(0, np.nan)

    # ── 시가총액 기반 ──
    df[f"mcap_change_1d_{c}"] = df[m].pct_change(1)
    df[f"mcap_change_7d_{c}"] = df[m].pct_change(7)

    return df


def make_onchain_features(df, coin):
    """온체인 공급량 파생변수"""
    c = f"supply_change_{coin}"
    circ = f"circ_{coin}"
    m    = f"market_cap_{coin}"

    if c in df.columns and circ in df.columns:
        df[f"mint_intensity_{coin}"] = df[c] / df[m].replace(0, np.nan)
        df[f"supply_growth_rate_{coin}"] = df[c] / df[circ].shift(1).replace(0, np.nan)

    return df


def make_macro_features(df):
    """거시경제 파생변수"""
    df["dxy_change_1d"]  = df["dxy"].pct_change(1)
    df["dxy_change_7d"]  = df["dxy"].pct_change(7)
    df["vix_change_1d"]  = df["vix"].pct_change(1)
    df["vix_spike"]      = (df["vix"] > 30).astype(int)
    df["rate_hike"]      = (df["federal_funds_rate"].diff() > 0).astype(int)
    df["btc_dom_change"] = df["btc_dominance"].pct_change(7)
    df["risk_off"]       = ((df["vix"] > 25) & (df["dxy_change_7d"] > 0)).astype(int)

    # BTC/ETH 시장
    df["btc_return_1d"]  = df["close_BTC"].pct_change(1)
    df["eth_return_1d"]  = df["close_ETH"].pct_change(1)
    df["btc_crash"]      = (df["btc_return_1d"] < -0.05).astype(int)
    df["btc_vol_7d"]     = df["btc_return_1d"].rolling(7).std()
    df["eth_vol_7d"]     = df["eth_return_1d"].rolling(7).std()
    df["crypto_stress"]  = ((df["btc_return_1d"] < -0.10) & (df["vix"] > 30)).astype(int)

    # FGI 파생
    df["fgi_change_1d"]  = df["fgi"].diff()
    df["extreme_fear"]   = (df["fgi"] < 20).astype(int)
    df["extreme_greed"]  = (df["fgi"] > 80).astype(int)

    # ── 추가 거시경제 파생 ──
    if "us_10y_yield" in df.columns:
        df["us_10y_change_1d"] = df["us_10y_yield"].diff()
        df["us_10y_change_7d"] = df["us_10y_yield"].diff(7)
        df["us_2y_change_1d"]  = df["us_2y_yield"].diff()
        df["yield_spread_change_7d"] = df["yield_spread_2s10s"].diff(7)
        # 역전 여부 (경기침체 신호)
        df["yield_curve_inverted"] = (df["yield_spread_2s10s"] < 0).astype(int)

    if "credit_spread" in df.columns:
        df["credit_spread_change_1d"] = df["credit_spread"].diff()
        df["credit_spread_change_7d"] = df["credit_spread"].diff(7)
        df["credit_spread_ma30"]      = df["credit_spread"].rolling(30).mean()
        # 신용 스프레드 급등 (30일 평균 대비 1.5배)
        cs_ma = df["credit_spread_ma30"].replace(0, np.nan)
        df["credit_stress"] = (df["credit_spread"] > cs_ma * 1.5).astype(int)

    if "m2_supply" in df.columns:
        df["m2_change_pct"] = df["m2_supply"].pct_change(4 * 7)  # 약 월간 변화율 (주간 데이터 ffill 기준)
        df["m2_yoy_approx"] = df["m2_supply"].pct_change(52 * 7)  # 약 연간 변화율

    if "fed_balance_sheet" in df.columns:
        df["fed_bs_change_pct"] = df["fed_balance_sheet"].pct_change(4 * 7)
        # 양적 긴축 여부 (Fed 대차대조표 축소)
        df["qt_signal"] = (df["fed_bs_change_pct"] < 0).astype(int)

    # 복합 유동성 스트레스 지표
    if all(c in df.columns for c in ["credit_stress", "vix_spike", "qt_signal"]):
        df["liquidity_stress"] = (
            df["credit_stress"] + df["vix_spike"] + df["qt_signal"]
        ).clip(upper=1)

    return df


def make_cross_coin_features(df):
    """코인 간 상대 변수 (USDT vs USDC 비교, DAI 연동성)"""
    # 스테이블코인 간 가격 괴리
    df["usdt_usdc_spread"]   = df["close_USDT"] - df["close_USDC"]
    df["usdt_dai_spread"]    = df["close_USDT"] - df["close_DAI"]

    # 거래량 집중도 (특정 코인으로 쏠림 → 고래 대리변수)
    total_vol = df["volume_USDT"] + df["volume_USDC"] + df["volume_DAI"]
    df["vol_share_USDT"]     = df["volume_USDT"] / total_vol.replace(0, np.nan)
    df["vol_share_USDC"]     = df["volume_USDC"] / total_vol.replace(0, np.nan)
    df["vol_share_DAI"]      = df["volume_DAI"]  / total_vol.replace(0, np.nan)

    # 시가총액 비율
    df["usdt_usdc_mcap_ratio"] = (
        df["market_cap_USDT"] / df["market_cap_USDC"].replace(0, np.nan)
    )

    # DAI 전용: ETH 가격과 연동 (암호 담보 특성)
    df["dai_eth_return_corr_30d"] = (
        df["return_1d_DAI"].rolling(30)
        .corr(df["eth_return_1d"])
    )

    return df


def make_sentiment_features(df):
    """심리/DeFi/가스 파생변수 생성"""
    # ── Google Trends 파생 ──
    if "gt_depeg" in df.columns:
        df["gt_depeg_total"] = (
            df.get("gt_depeg", 0) +
            df.get("gt_usdc_depeg", 0) +
            df.get("gt_dai_depeg", 0)
        )
        df["gt_fear_total"] = (
            df.get("gt_crypto_fear", 0) +
            df.get("gt_stablecoin_crash", 0) +
            df.get("gt_crypto_bank_run", 0)
        )
        df["gt_depeg_ma7"] = df["gt_depeg_total"].rolling(7).mean()
        ma7 = df["gt_depeg_ma7"].replace(0, np.nan)
        df["gt_depeg_spike"] = (df["gt_depeg_total"] > ma7 * 2).astype(int)
        df["gt_fear_ma7"] = df["gt_fear_total"].rolling(7).mean()

    # ── Gas Price 파생 ──
    if "gas_price_gwei" in df.columns:
        df["gas_price_ma7"] = df["gas_price_gwei"].rolling(7).mean()
        gas_ma7 = df["gas_price_ma7"].replace(0, np.nan)
        df["gas_price_spike"] = (df["gas_price_gwei"] > gas_ma7 * 2).astype(int)
        df["gas_price_change_1d"] = df["gas_price_gwei"].pct_change()

    # ── DeFi Lending TVL 파생 ──
    if "lending_tvl_total" in df.columns:
        df["lending_tvl_change_7d"] = df["lending_tvl_total"].pct_change(7)
        df["lending_tvl_drawdown"] = (
            df["lending_tvl_total"] / df["lending_tvl_total"].cummax() - 1
        )

    return df


# ── 5. 라그 변수 생성 (1일, 3일, 7일 전 값) ──────────────────────────────────

def make_lag_features(df, coin, lag_cols_base):
    """지정 컬럼에 대해 1/3/7일 라그 생성"""
    for col in lag_cols_base:
        full_col = f"{col}_{coin}" if f"{col}_{coin}" in df.columns else col
        if full_col not in df.columns:
            continue
        for lag in [1, 3, 7]:
            df[f"{full_col}_lag{lag}"] = df[full_col].shift(lag)
    return df


# ── 6. 코인별 데이터셋 분리 ───────────────────────────────────────────────────

def build_coin_df(df, coin):
    """특정 코인의 컬럼 + 공통 변수(거시, FGI, BTC/ETH) 추출"""
    # 해당 코인 컬럼
    coin_cols = [c for c in df.columns if coin in c]
    # 공통 컬럼
    common_cols = [
        "Date",
        "dxy", "vix", "federal_funds_rate", "btc_dominance",
        "dxy_change_1d", "dxy_change_7d", "vix_change_1d",
        "vix_spike", "rate_hike", "btc_dom_change", "risk_off",
        "btc_return_1d", "eth_return_1d", "btc_crash",
        "btc_vol_7d", "eth_vol_7d", "crypto_stress",
        "fgi", "fgi_class", "fgi_change_1d", "extreme_fear", "extreme_greed",
        "usdt_usdc_spread", "usdt_dai_spread",
        "vol_share_USDT", "vol_share_USDC", "vol_share_DAI",
        "usdt_usdc_mcap_ratio",
    ]
    # 추가 거시경제 변수
    macro_add_cols = [c for c in df.columns if any(
        c.startswith(p) for p in [
            "us_2y_", "us_3m_", "us_5y_", "us_10y_",
            "yield_spread", "yield_curve",
            "credit_spread", "credit_stress",
            "m2_", "fed_b", "qt_signal", "liquidity_stress",
        ]
    )]
    common_cols += macro_add_cols
    # Google Trends 변수
    gt_cols = [c for c in df.columns if c.startswith("gt_")]
    common_cols += gt_cols
    # Gas Price 변수
    gas_cols = [c for c in df.columns if "gas_price" in c or "eth_fees" in c or "eth_daily" in c]
    common_cols += gas_cols
    # DeFi Lending TVL 변수
    tvl_cols = [c for c in df.columns if c.startswith("tvl_") or c.startswith("lending_")]
    common_cols += tvl_cols
    # DAI 전용 추가
    if coin == "DAI":
        common_cols += ["dai_eth_return_corr_30d", "eth_vol_7d"]

    all_cols = [c for c in common_cols + coin_cols if c in df.columns]
    all_cols = list(dict.fromkeys(all_cols))  # 순서 유지 중복 제거
    return df[all_cols].copy()


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("1. 데이터 로딩")
    print("=" * 60)
    cmc       = load_cmc()
    macro     = load_macro()
    fgi       = load_fgi()
    onchain   = load_onchain()
    gt        = load_google_trends()
    gas       = load_gas_price()
    defi      = load_defi_protocols()
    macro_add = load_macro_additional()

    print("\n" + "=" * 60)
    print("2. 전체 병합")
    print("=" * 60)
    df = merge_all(cmc, macro, fgi, onchain, gt, gas, defi, macro_add)

    print("\n" + "=" * 60)
    print("3. 타깃 변수 생성")
    print("=" * 60)
    for coin in ["USDT", "USDC", "DAI"]:
        df = make_target(df, coin)

    print("\n" + "=" * 60)
    print("4. 파생변수 생성")
    print("=" * 60)
    for coin in ["USDT", "USDC", "DAI"]:
        df = make_features(df, coin)
        df = make_onchain_features(df, coin)
        print(f"  {coin} 파생변수 완료")

    df = make_macro_features(df)
    df = make_cross_coin_features(df)
    df = make_sentiment_features(df)
    print("  거시경제 / 코인 간 / 심리·DeFi 변수 완료")

    # 라그 변수
    LAG_BASE = ["return_1d", "vol_7d", "volume_ratio", "depeg_magnitude",
                "supply_change", "mcap_change_1d"]
    for coin in ["USDT", "USDC", "DAI"]:
        df = make_lag_features(df, coin, LAG_BASE)

    global_lag_cols = ["btc_return_1d", "eth_return_1d", "vix", "dxy"]
    # 신규 글로벌 라그 변수
    for col in ["gt_depeg_total", "gt_fear_total", "gas_price_gwei", "lending_tvl_total",
                "credit_spread", "us_10y_yield", "yield_spread_2s10s"]:
        if col in df.columns:
            global_lag_cols.append(col)
    df = make_lag_features(df, "", global_lag_cols)
    print("  라그 변수 완료")

    print(f"\n전체 변수 수: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("5. 코인별 데이터셋 분리 및 저장")
    print("=" * 60)

    # ── DAI 데이터셋 ──────────────────────────────────────────────────────────
    df_dai = build_coin_df(df, "DAI")
    # 초기 rolling 결측 (30일) 이후부터 유효
    df_dai = df_dai.dropna(subset=["rsi_14_DAI"]).reset_index(drop=True)
    dai_path = os.path.join(OUT_DIR, "df_dai.csv")
    df_dai.to_csv(dai_path, index=False)
    print(f"DAI: {df_dai.shape} → {dai_path}")
    print(f"  디페깅 이벤트: {df_dai['depeg_DAI'].sum()}개 ({df_dai['depeg_DAI'].mean()*100:.1f}%)")

    # ── USDT 데이터셋 ─────────────────────────────────────────────────────────
    df_usdt = build_coin_df(df, "USDT")
    df_usdt = df_usdt.dropna(subset=["rsi_14_USDT"]).reset_index(drop=True)
    usdt_path = os.path.join(OUT_DIR, "df_usdt.csv")
    df_usdt.to_csv(usdt_path, index=False)
    print(f"USDT: {df_usdt.shape} → {usdt_path}")
    print(f"  디페깅 이벤트: {df_usdt['depeg_USDT'].sum()}개 ({df_usdt['depeg_USDT'].mean()*100:.1f}%)")

    # ── USDC 데이터셋 ─────────────────────────────────────────────────────────
    df_usdc = build_coin_df(df, "USDC")
    df_usdc = df_usdc.dropna(subset=["rsi_14_USDC"]).reset_index(drop=True)
    usdc_path = os.path.join(OUT_DIR, "df_usdc.csv")
    df_usdc.to_csv(usdc_path, index=False)
    print(f"USDC: {df_usdc.shape} → {usdc_path}")
    print(f"  디페깅 이벤트: {df_usdc['depeg_USDC'].sum()}개 ({df_usdc['depeg_USDC'].mean()*100:.1f}%)")

    # ── 법정담보 통합 (USDT + USDC, long format) ─────────────────────────────
    df_usdt_l = df_usdt.copy(); df_usdt_l["coin"] = "USDT"
    df_usdc_l = df_usdc.copy(); df_usdc_l["coin"] = "USDC"

    # 컬럼명 통일 (USDT → coin, USDC → coin)
    def rename_coin_cols(d, old_coin):
        return d.rename(columns={c: c.replace(f"_{old_coin}", "_coin") for c in d.columns})

    df_usdt_l = rename_coin_cols(df_usdt_l, "USDT")
    df_usdc_l = rename_coin_cols(df_usdc_l, "USDC")

    # 공통 컬럼만 유지
    common = list(set(df_usdt_l.columns) & set(df_usdc_l.columns))
    df_fiat = pd.concat([df_usdt_l[common], df_usdc_l[common]], ignore_index=True)
    df_fiat = df_fiat.sort_values(["Date", "coin"]).reset_index(drop=True)

    fiat_path = os.path.join(OUT_DIR, "df_fiat.csv")
    df_fiat.to_csv(fiat_path, index=False)
    print(f"\n법정담보 통합 (USDT+USDC): {df_fiat.shape} → {fiat_path}")
    print(f"  디페깅 이벤트: {df_fiat['depeg_coin'].sum()}개 ({df_fiat['depeg_coin'].mean()*100:.1f}%)")
    print(f"  코인별: {df_fiat.groupby('coin')['depeg_coin'].sum().to_dict()}")

    # ── wide format 전체 저장 (나중에 통합 모델용) ────────────────────────────
    merged_path = os.path.join(OUT_DIR, "df_merged.csv")
    df.to_csv(merged_path, index=False)
    print(f"\n전체 wide format: {df.shape} → {merged_path}")

    # ── 최종 요약 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("완료 요약")
    print("=" * 60)
    for name, path, d in [
        ("df_fiat (USDT+USDC)", fiat_path, df_fiat),
        ("df_dai  (DAI)",       dai_path,  df_dai),
        ("df_usdt (USDT)",      usdt_path, df_usdt),
        ("df_usdc (USDC)",      usdc_path, df_usdc),
    ]:
        print(f"  {name}: {d.shape[0]}행 × {d.shape[1]}열")


if __name__ == "__main__":
    main()
