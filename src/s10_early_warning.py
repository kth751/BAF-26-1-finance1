"""
s10_early_warning.py — 3단계 조기경보 시스템

모델 예측 확률 P(depeg) 기반 경보 수준:
  - 경보 (Alert):   P >= threshold_alert   (높은 디페깅 위험)
  - 주의 (Caution): threshold_caution <= P < threshold_alert
  - 정상 (Normal):  P < threshold_caution

임계값 결정 (F2/F1 2단계 최적화):
  1. threshold_alert: F2-score 최적화 (Recall 중시, 놓치면 안 되는 위험)
  2. threshold_caution: F1-score 최적화 (Precision-Recall 균형, 사전 모니터링)
  F1은 Precision과 Recall을 균등 가중 → 자연스럽게 Alert보다 낮은 임계값

v2 최적 설정:
  - USDC: WeightOnly + XGB
  - DAI: ADASYN + XGB
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score, f1_score,
    roc_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN, SMOTE

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ML_DIR = os.path.join(PROJECT_DIR, "data", "ml")
PROC_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ml")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

V2_CONFIGS = {
    "USDC": {"technique": "WeightOnly", "model": "XGB"},
    "DAI": {"technique": "ADASYN", "model": "XGB"},
}


def fbeta(y_true, y_pred, beta=2.0):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    if p + r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r)


def load_data(coin):
    X = pd.read_csv(os.path.join(ML_DIR, f"v2_X_{coin.lower()}.csv"))
    Y = pd.read_csv(os.path.join(ML_DIR, f"v2_Y_{coin.lower()}.csv"), header=None).squeeze()
    X = X.replace([np.inf, -np.inf], np.nan)

    # 날짜 매핑: df_{coin}.csv에서 Date 가져오기
    df_raw = pd.read_csv(os.path.join(PROC_DIR, f"df_{coin.lower()}.csv"), parse_dates=["Date"])
    # Y = shift(-1) → 마지막 행 제거 → dates[:-1]
    dates = df_raw["Date"].values[:len(Y)]

    return X.values, Y.values, X.columns.tolist(), dates


def build_and_sample(X_train, Y_train, coin):
    cfg = V2_CONFIGS[coin]
    technique = cfg["technique"]

    if technique in ("WeightOnly", "Baseline", None):
        return X_train, Y_train

    n_pos = int(Y_train.sum())
    k = min(5, n_pos - 1)
    if k < 1:
        return X_train, Y_train

    if technique == "ADASYN":
        sampler = ADASYN(sampling_strategy=0.5, n_neighbors=k, random_state=RANDOM_STATE)
        X_res, Y_res = sampler.fit_resample(X_train, Y_train)
    elif technique.startswith("SMOTE"):
        sampler = SMOTE(sampling_strategy=0.5, k_neighbors=k, random_state=RANDOM_STATE)
        X_res, Y_res = sampler.fit_resample(X_train, Y_train)
    else:
        X_res, Y_res = X_train, Y_train

    return X_res, Y_res


def build_model(coin, n_pos, n_neg):
    cfg = V2_CONFIGS[coin]
    use_weight = (cfg["technique"] == "WeightOnly")

    if cfg["model"] == "RF":
        return RandomForestClassifier(
            n_estimators=300, max_depth=8,
            class_weight="balanced" if use_weight else None,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    else:
        spw = (n_neg / n_pos) if (use_weight and n_pos > 0) else 1
        return XGBClassifier(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
        )


# ── 1. CV 기반 out-of-fold 예측 확률 ──

def get_oof_probabilities(X, Y, coin):
    """StratifiedKFold로 out-of-fold 예측 확률 생성 (data leakage 최소화)"""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(Y))
    imp = SimpleImputer(strategy="median")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        X_train = imp.fit_transform(X_train)
        X_test = imp.transform(X_test)

        X_res, Y_res = build_and_sample(X_train, Y_train, coin)
        n_pos = int((Y_res == 1).sum())
        n_neg = int((Y_res == 0).sum())

        model = build_model(coin, n_pos, n_neg)
        model.fit(X_res, Y_res)
        oof_proba[test_idx] = model.predict_proba(X_test)[:, 1]

    return oof_proba


# ── 2. 임계값 최적화 ──

def find_thresholds(Y, proba, coin):
    """
    2단계 임계값 최적화:
      threshold_alert:   F2 최적화 (Recall 중시 → 높은 확률에서 경보)
      threshold_caution: Youden's J 최적화 (Sensitivity + Specificity - 1 최대화)

    Youden's J는 ROC 기반 지표로 민감도와 특이도의 균형점을 찾음.
    F2보다 구조적으로 낮은 임계값이 도출되어 사전 모니터링에 적합.
    의학 진단 분야에서 표준적으로 사용되는 최적 cut-off 결정법 (Youden, 1950).
    """

    # F2 최적 threshold → Alert
    best_f2 = 0
    best_t_alert = 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        y_pred = (proba >= t).astype(int)
        f2 = fbeta(Y, y_pred, beta=2.0)
        if f2 > best_f2:
            best_f2 = f2
            best_t_alert = t

    # Youden's J 최적 threshold → Caution
    fpr, tpr, roc_thresholds = roc_curve(Y, proba)
    j_scores = tpr - fpr  # Youden's J = Sensitivity + Specificity - 1 = TPR - FPR
    best_j_idx = np.argmax(j_scores)
    best_t_caution = roc_thresholds[best_j_idx]
    best_j = j_scores[best_j_idx]

    # Caution이 Alert 이상이면 Alert의 절반으로 (구조적 보장)
    if best_t_caution >= best_t_alert:
        # J가 최대인 지점이 Alert 이상 → Alert 미만에서 J 최대 재탐색
        below_alert = roc_thresholds < best_t_alert
        if below_alert.any():
            j_below = j_scores.copy()
            j_below[~below_alert] = -1
            best_j_idx = np.argmax(j_below)
            best_t_caution = roc_thresholds[best_j_idx]
            best_j = j_scores[best_j_idx]
        else:
            best_t_caution = best_t_alert * 0.5

    # 성능 리포트
    for label, t, metric_name, metric_val in [
        ("Alert", best_t_alert, "F2", best_f2),
        ("Caution", best_t_caution, "Youden's J", best_j),
    ]:
        y_pred = (proba >= t).astype(int)
        r = recall_score(Y, y_pred, zero_division=0)
        p = precision_score(Y, y_pred, zero_division=0)
        f2 = fbeta(Y, y_pred)
        print(f"    {label} (t={t:.3f}): {metric_name}={metric_val:.3f}, Recall={r:.3f}, Precision={p:.3f}, F2={f2:.3f}")

    return best_t_alert, best_t_caution, best_f2, best_j


# ── 3. 전체 데이터 최종 모델 예측 ──

def get_full_probabilities(X, Y, coin):
    """전체 데이터로 학습 후 각 날짜 예측 확률 (시각화용)"""
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    X_res, Y_res = build_and_sample(X_imp, Y, coin)
    n_pos = int((Y_res == 1).sum())
    n_neg = int((Y_res == 0).sum())

    model = build_model(coin, n_pos, n_neg)
    model.fit(X_res, Y_res)
    proba = model.predict_proba(X_imp)[:, 1]

    return proba, model


# ── 4. 시각화 ──

def plot_warning_timeline(dates, proba, Y, t_alert, t_caution, coin):
    """전체 기간 경보 타임라인"""
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={"height_ratios": [3, 1]})

    dates_dt = pd.to_datetime(dates)

    # 상단: 예측 확률 + 임계값
    ax1 = axes[0]
    ax1.fill_between(dates_dt, proba, alpha=0.3, color="#3498DB", label="P(depeg)")
    ax1.plot(dates_dt, proba, linewidth=0.5, color="#2C3E50", alpha=0.7)
    ax1.axhline(t_alert, color="#E74C3C", linestyle="--", linewidth=1.5, label=f"Alert ({t_alert:.2f})")
    ax1.axhline(t_caution, color="#F39C12", linestyle="--", linewidth=1.5, label=f"Caution ({t_caution:.2f})")

    # 실제 디페깅 마킹
    depeg_mask = Y == 1
    ax1.scatter(dates_dt[depeg_mask], proba[depeg_mask],
                color="#E74C3C", s=15, zorder=5, label=f"Actual depeg (n={depeg_mask.sum()})")

    ax1.set_ylabel("P(depeg)", fontsize=12)
    ax1.set_title(f"{coin} — Early Warning System (3-Level)", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xlim(dates_dt[0], dates_dt[-1])
    ax1.set_ylim(-0.02, 1.02)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # 하단: 경보 수준 색상 바
    ax2 = axes[1]
    level = np.zeros(len(proba))
    level[proba >= t_caution] = 1  # 주의
    level[proba >= t_alert] = 2    # 경보

    colors_map = {0: "#2ECC71", 1: "#F39C12", 2: "#E74C3C"}
    for i in range(len(dates_dt) - 1):
        ax2.axvspan(dates_dt[i], dates_dt[i+1],
                    color=colors_map[level[i]], alpha=0.7)

    legend_elements = [
        Patch(facecolor="#2ECC71", alpha=0.7, label="Normal"),
        Patch(facecolor="#F39C12", alpha=0.7, label="Caution"),
        Patch(facecolor="#E74C3C", alpha=0.7, label="Alert"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax2.set_ylabel("Warning Level", fontsize=11)
    ax2.set_yticks([])
    ax2.set_xlim(dates_dt[0], dates_dt[-1])
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"early_warning_{coin.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  타임라인: {path}")


def plot_warning_zoom(dates, proba, Y, t_alert, t_caution, coin, period_name, start, end):
    """특정 기간 확대 플롯"""
    dates_dt = pd.to_datetime(dates)
    mask = (dates_dt >= start) & (dates_dt <= end)

    if mask.sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    d = dates_dt[mask]
    p = proba[mask]
    y = Y[mask]

    # 배경 색상
    for i in range(len(d) - 1):
        if p[i] >= t_alert:
            color = "#FADBD8"
        elif p[i] >= t_caution:
            color = "#FDEBD0"
        else:
            color = "#D5F5E3"
        ax.axvspan(d[i], d[i+1], color=color, alpha=0.6)

    ax.plot(d, p, color="#2C3E50", linewidth=1.5, marker="o", markersize=2)
    ax.axhline(t_alert, color="#E74C3C", linestyle="--", linewidth=1.5, label=f"Alert ({t_alert:.2f})")
    ax.axhline(t_caution, color="#F39C12", linestyle="--", linewidth=1.5, label=f"Caution ({t_caution:.2f})")

    depeg_mask = y == 1
    if depeg_mask.sum() > 0:
        ax.scatter(d[depeg_mask], p[depeg_mask], color="#E74C3C", s=40, zorder=5, label="Actual depeg")

    ax.set_title(f"{coin} — {period_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("P(depeg)")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    safe_name = period_name.replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(OUT_DIR, f"early_warning_{coin.lower()}_{safe_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  확대: {path}")


def print_warning_stats(dates, proba, Y, t_alert, t_caution, coin):
    """경보 통계 요약"""
    dates_dt = pd.to_datetime(dates)
    n = len(Y)

    alert = proba >= t_alert
    caution = (proba >= t_caution) & (proba < t_alert)
    normal = proba < t_caution

    print(f"\n  [{coin}] 경보 분포:")
    print(f"    정상 (Normal):  {normal.sum():5d}일 ({normal.sum()/n*100:.1f}%)")
    print(f"    주의 (Caution): {caution.sum():5d}일 ({caution.sum()/n*100:.1f}%)")
    print(f"    경보 (Alert):   {alert.sum():5d}일 ({alert.sum()/n*100:.1f}%)")

    # 디페깅 이벤트 포착률
    depeg = Y == 1
    n_depeg = depeg.sum()
    if n_depeg > 0:
        caught_alert = (alert & depeg).sum()
        caught_caution = (caution & depeg).sum()
        caught_any = ((alert | caution) & depeg).sum()
        missed = n_depeg - caught_any

        print(f"\n  디페깅 포착률 ({n_depeg}건):")
        print(f"    경보에서 포착: {caught_alert:3d}건 ({caught_alert/n_depeg*100:.1f}%)")
        print(f"    주의에서 포착: {caught_caution:3d}건 ({caught_caution/n_depeg*100:.1f}%)")
        print(f"    합계 포착:     {caught_any:3d}건 ({caught_any/n_depeg*100:.1f}%)")
        print(f"    미포착 (정상): {missed:3d}건 ({missed/n_depeg*100:.1f}%)")

    # 오경보율
    non_depeg = Y == 0
    false_alert = (alert & non_depeg).sum()
    false_caution = (caution & non_depeg).sum()
    print(f"\n  오경보:")
    print(f"    경보 오경보: {false_alert:3d}일 (전체 경보 {alert.sum()}일 중)")
    print(f"    주의 오경보: {false_caution:3d}일 (전체 주의 {caution.sum()}일 중)")

    # 사전 경보: 디페깅 발생 N일 전에 주의/경보가 있었는지
    depeg_starts = []
    in_depeg = False
    for i in range(len(Y)):
        if Y[i] == 1 and not in_depeg:
            depeg_starts.append(i)
            in_depeg = True
        elif Y[i] == 0:
            in_depeg = False

    print(f"\n  사전 경보 (디페깅 에피소드 {len(depeg_starts)}건):")
    for si in depeg_starts:
        # 이전 7일 내 주의/경보 여부
        lookback = 7
        start_idx = max(0, si - lookback)
        pre_window = proba[start_idx:si]
        pre_alert = (pre_window >= t_alert).any()
        pre_caution = (pre_window >= t_caution).any()
        date_str = pd.Timestamp(dates_dt[si]).strftime("%Y-%m-%d")

        if pre_alert:
            lead = "Alert"
        elif pre_caution:
            lead = "Caution"
        else:
            lead = "None"

        # 사전 경보 리드 타임
        lead_days = 0
        if pre_caution:
            for j in range(si - 1, start_idx - 1, -1):
                if proba[j] >= t_caution:
                    lead_days = si - j
                    break

        print(f"    {date_str}: 사전 {lookback}일 내 {lead:8s} (리드타임: {lead_days}일)")


def main():
    all_results = []

    for coin in ["USDC", "DAI"]:
        cfg = V2_CONFIGS[coin]
        print(f"\n{'='*65}")
        print(f"  {coin} 조기경보 시스템 ({cfg['technique']}+{cfg['model']})")
        print(f"{'='*65}")

        X, Y, feat_names, dates = load_data(coin)
        print(f"  데이터: {X.shape}, depeg={int(Y.sum())}건")

        # 1) OOF 예측 확률 (임계값 결정용)
        print(f"\n  [Step 1] Out-of-Fold 예측 확률 계산...")
        oof_proba = get_oof_probabilities(X, Y, coin)

        # 2) 임계값 결정
        print(f"\n  [Step 2] 임계값 최적화...")
        t_alert, t_caution, best_f2, best_j = find_thresholds(Y, oof_proba, coin)
        print(f"    -> Alert: {t_alert:.3f} (F2={best_f2:.3f}), Caution: {t_caution:.3f} (J={best_j:.3f})")

        # 3) 전체 모델 예측 (시각화용)
        print(f"\n  [Step 3] 전체 데이터 예측...")
        full_proba, model = get_full_probabilities(X, Y, coin)

        # 4) 경보 통계
        print_warning_stats(dates, full_proba, Y, t_alert, t_caution, coin)

        # 5) 시각화
        print(f"\n  [Step 4] 시각화...")
        plot_warning_timeline(dates, full_proba, Y, t_alert, t_caution, coin)

        # 주요 이벤트 확대
        zoom_periods = [
            ("COVID_2020-03", "2020-02-15", "2020-04-15"),
            ("DeFi_2020-08", "2020-07-15", "2020-09-30"),
            ("SVB_2023-03", "2023-02-25", "2023-03-25"),
        ]
        for name, start, end in zoom_periods:
            plot_warning_zoom(dates, full_proba, Y, t_alert, t_caution, coin, name, start, end)

        # 결과 저장
        result_df = pd.DataFrame({
            "Date": dates,
            "Y_actual": Y,
            "P_depeg_oof": oof_proba,
            "P_depeg_full": full_proba,
            "level": np.where(full_proba >= t_alert, "Alert",
                     np.where(full_proba >= t_caution, "Caution", "Normal")),
        })
        result_df.to_csv(os.path.join(ML_DIR, f"early_warning_{coin.lower()}.csv"), index=False)

        all_results.append({
            "coin": coin,
            "t_alert": t_alert,
            "t_caution": t_caution,
            "f2_oof": best_f2,
            "youden_j": best_j,
        })

    # 임계값 요약 저장
    pd.DataFrame(all_results).to_csv(
        os.path.join(ML_DIR, "early_warning_thresholds.csv"), index=False
    )

    print(f"\n{'='*65}")
    print(f"  조기경보 시스템 구축 완료")
    print(f"{'='*65}")
    for r in all_results:
        print(f"  {r['coin']}: Alert >= {r['t_alert']:.3f} (F2={r['f2_oof']:.3f}), "
              f"Caution >= {r['t_caution']:.3f} (J={r['youden_j']:.3f})")
    print(f"\n  결과: data/ml/early_warning_*.csv")
    print(f"  시각화: outputs/ml/early_warning_*.png")


if __name__ == "__main__":
    main()
