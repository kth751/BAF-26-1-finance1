"""
s9_shap_v3.py — v3 데이터(텍스트 변수 포함) SHAP 분석

v3 최적 설정:
- USDC: ADASYN + XGB (F2=0.894)
- DAI: SMOTE_05 + RF (F2=0.887)

TreeExplainer 사용 (XGB/RF 모두 tree 기반 → 빠름)
"""

import pandas as pd
import numpy as np
import os
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

# v3 최적 설정
V3_CONFIGS = {
    "USDC": {"technique": "ADASYN", "model": "XGB"},
    "DAI": {"technique": "SMOTE_05", "model": "RF"},
}


def load_data_v3(coin):
    X = pd.read_csv(os.path.join(ML_DIR, f"v3_X_{coin.lower()}.csv"))
    Y = pd.read_csv(os.path.join(ML_DIR, f"v3_Y_{coin.lower()}.csv"), header=None).squeeze()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.values, Y.values, X.columns.tolist()


def train_full_model(coin):
    cfg = V3_CONFIGS[coin]
    technique = cfg["technique"]
    model_name = cfg["model"]

    X, Y, feat_names = load_data_v3(coin)

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    n_pos = int(Y.sum())
    n_neg = int((Y == 0).sum())

    # Sampling
    k = min(5, n_pos - 1)
    if technique == "ADASYN":
        sampler = ADASYN(sampling_strategy=0.5, n_neighbors=k, random_state=RANDOM_STATE)
        X_train, Y_train = sampler.fit_resample(X_imp, Y)
    elif technique.startswith("SMOTE"):
        sampler = SMOTE(sampling_strategy=0.5, k_neighbors=k, random_state=RANDOM_STATE)
        X_train, Y_train = sampler.fit_resample(X_imp, Y)
    else:
        X_train, Y_train = X_imp, Y

    n_pos_r = int((Y_train == 1).sum())
    n_neg_r = int((Y_train == 0).sum())

    # Model
    if model_name == "RF":
        model = RandomForestClassifier(
            n_estimators=300, max_depth=8,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    else:
        spw = n_neg_r / n_pos_r if n_pos_r > 0 else 1
        model = XGBClassifier(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
        )

    model.fit(X_train, Y_train)
    print(f"  [{coin}] {technique}+{model_name} 학습: {X_imp.shape[0]}→{X_train.shape[0]}행")

    return model, X_imp, Y, feat_names


def compute_shap(model, X, feat_names, coin):
    model_type = V3_CONFIGS[coin]["model"]
    print(f"  [{coin}] SHAP 계산 중 ({model_type})...")

    X_df = pd.DataFrame(X, columns=feat_names)

    if model_type == "XGB":
        # XGBoost 3.x + shap 0.49 TreeExplainer 호환 불가
        # XGBoost 자체 predict(pred_contribs=True) 사용
        import xgboost as xgb
        dmat = xgb.DMatrix(X_df, feature_names=feat_names)
        contribs = model.get_booster().predict(dmat, pred_contribs=True)
        # contribs shape: (n_samples, n_features+1), 마지막 열 = bias
        shap_values = contribs[:, :-1]
        expected_value = contribs[0, -1]
    else:
        # RF → TreeExplainer 사용 가능
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # list [class0, class1] 또는 3D array (n, features, 2) 처리
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

    # CSV 저장
    df_shap = pd.DataFrame(shap_values, columns=feat_names)
    df_shap.to_csv(os.path.join(ML_DIR, f"shap_values_v3_{coin.lower()}.csv"), index=False)
    print(f"  [{coin}] SHAP values 저장 완료")

    return shap_values, expected_value


def plot_summary_beeswarm(shap_values, X, feat_names, coin):
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, feature_names=feat_names,
                      max_display=20, show=False)
    plt.title(f"{coin} — SHAP Summary v3 (Beeswarm)")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"shap_v3_summary_{coin.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    {path}")


def plot_bar_global(shap_values, X, feat_names, coin):
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feat_names,
                      plot_type="bar", max_display=20, show=False)
    plt.title(f"{coin} — SHAP Feature Importance v3 (Top 20)")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"shap_v3_bar_{coin.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    {path}")


def plot_dependence_top_n(shap_values, X, feat_names, coin, n=5):
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:n]

    for rank, idx in enumerate(top_idx):
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(idx, shap_values, X, feature_names=feat_names,
                             ax=ax, show=False)
        ax.set_title(f"{coin} — Dependence: {feat_names[idx]} (#{rank+1})")
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"shap_v3_dep_{coin.lower()}_top{rank+1}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"    Dependence plot: top {n}개")


def plot_force_depeg_samples(shap_values, X, Y, feat_names, coin, n_samples=5):
    depeg_idx = np.where(Y == 1)[0]
    if len(depeg_idx) == 0:
        print(f"    [{coin}] 디페깅 샘플 없음 → 스킵")
        return

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

    plt.suptitle(f"{coin} — Force Plot v3 (Top Depeg Samples)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"shap_v3_force_{coin.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Force plot: {len(sample_indices)}개 샘플")


def plot_coin_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, (coin, data) in zip(axes, results.items()):
        sv = data["shap_values"]
        feat_names = data["feat_names"]
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:15]

        y_pos = range(len(top_idx))
        ax.barh(y_pos, mean_abs[top_idx],
                color="#DD8452" if coin == "USDC" else "#4C72B0", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=8)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{coin} — Top 15 ({V3_CONFIGS[coin]['technique']}+{V3_CONFIGS[coin]['model']})")
        ax.invert_yaxis()

    plt.suptitle("USDC vs DAI — SHAP Feature Importance v3", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "shap_v3_coin_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  비교 플롯: {path}")


def plot_text_feature_highlight(results):
    """텍스트 변수가 전체 변수 중 어느 위치에 있는지 하이라이트"""
    text_prefixes = ("kw_", "risk_", "positive_", "topic_", "emb_")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, (coin, data) in zip(axes, results.items()):
        sv = data["shap_values"]
        feat_names = data["feat_names"]
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:20]

        names = [feat_names[i] for i in top_idx]
        vals = mean_abs[top_idx]
        is_text = [any(n.startswith(p) for p in text_prefixes) for n in names]
        colors = ["#E74C3C" if t else "#3498DB" for t in is_text]

        ax.barh(range(len(top_idx)), vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{coin} — Text Features Highlighted (red)")
        ax.invert_yaxis()

    plt.suptitle("v3 SHAP — Text vs Numeric Feature Comparison", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "shap_v3_text_highlight.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  텍스트 하이라이트: {path}")


def main():
    results = {}

    for coin in ["USDC", "DAI"]:
        cfg = V3_CONFIGS[coin]
        print(f"\n{'='*60}")
        print(f"  {coin} SHAP v3 ({cfg['technique']}+{cfg['model']})")
        print(f"{'='*60}")

        model, X, Y, feat_names = train_full_model(coin)
        shap_values, expected_value = compute_shap(model, X, feat_names, coin)

        print(f"\n  시각화 생성...")
        plot_summary_beeswarm(shap_values, X, feat_names, coin)
        plot_bar_global(shap_values, X, feat_names, coin)
        plot_dependence_top_n(shap_values, X, feat_names, coin, n=5)
        plot_force_depeg_samples(shap_values, X, Y, feat_names, coin, n_samples=5)

        results[coin] = {"shap_values": shap_values, "feat_names": feat_names}

        # 주요 변수 출력
        mean_abs = np.abs(shap_values).mean(axis=0)
        top10_idx = np.argsort(mean_abs)[::-1][:10]
        print(f"\n  [{coin}] Top 10 변수:")
        for rank, idx in enumerate(top10_idx):
            marker = " *TEXT*" if any(feat_names[idx].startswith(p) for p in
                                     ("kw_", "risk_", "positive_", "topic_", "emb_")) else ""
            print(f"    {rank+1:2d}. {feat_names[idx]:35s} |SHAP|={mean_abs[idx]:.4f}{marker}")

    print(f"\n{'='*60}")
    print(f"  코인 비교 + 텍스트 하이라이트")
    print(f"{'='*60}")
    plot_coin_comparison(results)
    plot_text_feature_highlight(results)

    print(f"\n완료! 출력: outputs/ml/shap_v3_*.png")


if __name__ == "__main__":
    main()
