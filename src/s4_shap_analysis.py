"""
s4_shap_analysis.py — SHAP 해석 분석

best_config.json 기반 최적 모델로 전체 데이터 학습 후 SHAP 분석.
CV 성능은 s2에서 검증 완료 → 여기서는 해석 목적으로 전체 데이터 사용.

입력: data/ml/best_config.json, v2_X/Y_{coin}.csv
출력: outputs/ml/shap_*.png, data/ml/shap_values_{coin}.csv
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN, SMOTE

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ML_DIR = os.path.join(PROJECT_DIR, "data", "ml")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs", "ml")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ── 재사용 함수 (s2에서 복사) ──

def load_data(coin):
    X = pd.read_csv(os.path.join(ML_DIR, f"v2_X_{coin.lower()}.csv"))
    Y = pd.read_csv(os.path.join(ML_DIR, f"v2_Y_{coin.lower()}.csv"), header=None).squeeze()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.values, Y.values, X.columns.tolist()


def build_model(config, n_pos=None, n_neg=None):
    model_type = config["model"]
    use_weight = config.get("weight", False)

    if model_type == "RF":
        return RandomForestClassifier(
            n_estimators=300, max_depth=8,
            class_weight="balanced" if use_weight else None,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    elif model_type == "XGB":
        spw = (n_neg / n_pos) if (use_weight and n_pos and n_pos > 0) else 1
        return XGBClassifier(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss", base_score=0.5,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
        )


def build_sampler(name, n_minority, ratio=0.5):
    if name is None or name == "WeightOnly" or name == "Baseline":
        return None

    k = min(5, n_minority - 1)
    if k < 1:
        return None

    if name == "ADASYN":
        return ADASYN(sampling_strategy=ratio, n_neighbors=k, random_state=RANDOM_STATE)
    elif name == "SMOTE" or name.startswith("SMOTE_"):
        return SMOTE(sampling_strategy=ratio, k_neighbors=k, random_state=RANDOM_STATE)
    return None


def technique_to_config(technique, model_name):
    """best_config의 technique/model → build_model용 config 변환"""
    use_weight = (technique == "WeightOnly")
    return {"model": model_name, "weight": use_weight}


# ── SHAP 분석 ──

def train_full_model(coin, technique, model_name):
    """전체 데이터로 모델 학습"""
    X, Y, feat_names = load_data(coin)

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    n_pos = int(Y.sum())
    n_neg = int((Y == 0).sum())

    sampler = build_sampler(technique, n_pos)
    if sampler is not None:
        X_train, Y_train = sampler.fit_resample(X_imp, Y)
    else:
        X_train, Y_train = X_imp, Y

    n_pos_res = int((Y_train == 1).sum())
    n_neg_res = int((Y_train == 0).sum())

    config = technique_to_config(technique, model_name)
    model = build_model(config, n_pos=n_pos_res, n_neg=n_neg_res)
    model.fit(X_train, Y_train)

    print(f"  [{coin}] 학습 완료: {technique}+{model_name}, "
          f"데이터={X_imp.shape[0]}→{X_train.shape[0]}행")

    return model, X_imp, Y, feat_names


def compute_shap(model, X, feat_names, coin):
    """SHAP values 계산 — XGBoost pred_contribs (가장 빠르고 호환 안정적)"""
    print(f"  [{coin}] SHAP 계산 중 ({X.shape[0]} samples x {X.shape[1]} features)...")

    import xgboost as xgb
    X_df = pd.DataFrame(X, columns=feat_names)
    dmatrix = xgb.DMatrix(X_df, feature_names=feat_names)

    # pred_contribs: 마지막 열이 base_value, 나머지가 SHAP values
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_values = contribs[:, :-1]
    expected_value = contribs[0, -1]

    # CSV 저장
    df_shap = pd.DataFrame(shap_values, columns=feat_names)
    df_shap.to_csv(os.path.join(ML_DIR, f"shap_values_{coin.lower()}.csv"), index=False)
    print(f"  [{coin}] SHAP values 저장: shap_values_{coin.lower()}.csv")
    print(f"  [{coin}] base_value={expected_value:.4f}")

    return shap_values, None, expected_value


def plot_summary_beeswarm(shap_values, X, feat_names, coin):
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, feature_names=feat_names,
                      max_display=20, show=False)
    plt.title(f"{coin} — SHAP Summary (Beeswarm)")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"shap_summary_{coin.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    저장: {path}")


def plot_bar_global(shap_values, X, feat_names, coin):
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feat_names,
                      plot_type="bar", max_display=20, show=False)
    plt.title(f"{coin} — SHAP Feature Importance (Top 20)")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"shap_bar_{coin.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    저장: {path}")


def plot_dependence_top_n(shap_values, X, feat_names, coin, n=5):
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:n]

    for rank, idx in enumerate(top_idx):
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(idx, shap_values, X, feature_names=feat_names,
                             ax=ax, show=False)
        ax.set_title(f"{coin} — Dependence: {feat_names[idx]} (#{rank+1})")
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"shap_dependence_{coin.lower()}_top{rank+1}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"    Dependence plot 저장: top {n}개")


def plot_force_depeg_samples(shap_values, X, Y, feat_names, coin, n_samples=5):
    depeg_idx = np.where(Y == 1)[0]
    if len(depeg_idx) == 0:
        print(f"    [{coin}] 디페깅 샘플 없음 → force plot 스킵")
        return

    # SHAP value 절대합이 큰 디페깅 샘플 선택
    abs_shap_sum = np.abs(shap_values[depeg_idx]).sum(axis=1)
    top_local = np.argsort(abs_shap_sum)[::-1][:n_samples]
    sample_indices = depeg_idx[top_local]

    fig, axes = plt.subplots(len(sample_indices), 1,
                             figsize=(16, 3 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = [axes]

    for i, (ax, si) in enumerate(zip(axes, sample_indices)):
        top_k = 10
        sv = shap_values[si]
        top_feat_idx = np.argsort(np.abs(sv))[::-1][:top_k]
        colors = ["#FF4136" if sv[j] > 0 else "#2196F3" for j in top_feat_idx]

        ax.barh(range(top_k), sv[top_feat_idx], color=colors, alpha=0.8)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f"{feat_names[j]}={X[si, j]:.3f}" for j in top_feat_idx],
                           fontsize=7)
        ax.set_xlabel("SHAP value")
        ax.set_title(f"Sample #{si} (depeg=1)", fontsize=9)
        ax.invert_yaxis()

    plt.suptitle(f"{coin} — Force Plot (Top Depeg Samples)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"shap_force_{coin.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Force plot 저장: {n_samples}개 샘플")


def plot_coin_comparison(results):
    """두 코인의 SHAP importance 비교"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, (coin, data) in zip(axes, results.items()):
        sv = data["shap_values"]
        feat_names = data["feat_names"]
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:15]

        y_pos = range(len(top_idx))
        ax.barh(y_pos, mean_abs[top_idx], color="#DD8452" if coin == "USDC" else "#4C72B0",
                alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=8)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{coin} — Top 15 Features")
        ax.invert_yaxis()

    plt.suptitle("USDC vs DAI — SHAP Feature Importance Comparison", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "shap_coin_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  비교 플롯 저장: {path}")


def main():
    with open(os.path.join(ML_DIR, "best_config.json"), "r", encoding="utf-8") as f:
        best_config = json.load(f)

    results = {}

    for coin in ["USDC", "DAI"]:
        print(f"\n{'='*60}")
        print(f"  {coin} SHAP 분석")
        print(f"{'='*60}")

        bc = best_config[coin]["best_f2"]
        technique = bc["technique"]
        model_name = bc["model"]
        print(f"  설정: {technique}+{model_name} (F2={bc['f2']})")

        model, X, Y, feat_names = train_full_model(coin, technique, model_name)
        shap_values, explainer, expected_value = compute_shap(model, X, feat_names, coin)

        print(f"\n  시각화 생성 중...")
        plot_summary_beeswarm(shap_values, X, feat_names, coin)
        plot_bar_global(shap_values, X, feat_names, coin)
        plot_dependence_top_n(shap_values, X, feat_names, coin, n=5)
        plot_force_depeg_samples(shap_values, X, Y, feat_names, coin, n_samples=5)

        results[coin] = {
            "shap_values": shap_values,
            "feat_names": feat_names,
        }

    print(f"\n{'='*60}")
    print(f"  USDC vs DAI 비교")
    print(f"{'='*60}")
    plot_coin_comparison(results)

    print(f"\n{'='*60}")
    print(f"  SHAP 분석 완료")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
