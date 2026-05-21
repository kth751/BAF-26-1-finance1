"""
s2_imbalance_experiment.py — 불균형 기법 체계적 비교 실험

13 기법 × 2 모델(RF/XGBoost) × 2 코인(USDC/DAI) × StratifiedKFold(5-fold)

CV 전략:
  디페깅 이벤트가 전체 관측치의 1.4~5.5%에 불과하며, 시계열 기반 분할 시
  특정 구간(2020년)에 이벤트가 편중되어 모델 학습·평가가 불가능하다.
  이에 Stratified K-Fold Cross Validation을 채택하여 각 fold에 디페깅 이벤트를
  균등 배분한다. 시계열 순서 미반영에 따른 data leakage는 연구의 한계로 명시한다.

입력: data/ml/v2_X_{coin}.csv, v2_Y_{coin}.csv
출력: data/ml/imbalance_results.csv
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
    precision_recall_curve, f1_score, recall_score, precision_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ML_DIR = os.path.join(PROJECT_DIR, "data", "ml")

RANDOM_STATE = 42
N_SPLITS = 5


def load_data(coin):
    X = pd.read_csv(os.path.join(ML_DIR, f"v2_X_{coin.lower()}.csv"))
    Y = pd.read_csv(os.path.join(ML_DIR, f"v2_Y_{coin.lower()}.csv"), header=None).squeeze()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.values, Y.values, X.columns.tolist()


def fbeta_score(y_true, y_pred, beta):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    if p + r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r)


def find_best_threshold(y_true, y_prob, beta=1.0):
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    eps = 1e-8
    fb = ((1 + beta**2) * prec[:-1] * rec[:-1]
          / (beta**2 * prec[:-1] + rec[:-1] + eps))
    best_idx = fb.argmax()
    return thresholds[best_idx]


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
        "f2": fbeta_score(y_true, y_pred, beta=2.0),
    }


def get_experiments():
    configs = []

    # 1. Baseline
    configs.append({"name": "Baseline", "sampler": None, "model": "RF", "weight": False, "thresh_beta": None})
    configs.append({"name": "Baseline", "sampler": None, "model": "XGB", "weight": False, "thresh_beta": None})

    # 2. WeightOnly
    configs.append({"name": "WeightOnly", "sampler": None, "model": "RF", "weight": True, "thresh_beta": None})
    configs.append({"name": "WeightOnly", "sampler": None, "model": "XGB", "weight": True, "thresh_beta": None})

    # 3~5. SMOTE ratios
    for ratio in [0.3, 0.5, 0.7]:
        label = f"SMOTE_{int(ratio*10):02d}"
        configs.append({"name": label, "sampler": "SMOTE", "ratio": ratio, "model": "RF", "weight": False, "thresh_beta": None})
        configs.append({"name": label, "sampler": "SMOTE", "ratio": ratio, "model": "XGB", "weight": False, "thresh_beta": None})

    # 6. ADASYN
    configs.append({"name": "ADASYN", "sampler": "ADASYN", "model": "RF", "weight": False, "thresh_beta": None})
    configs.append({"name": "ADASYN", "sampler": "ADASYN", "model": "XGB", "weight": False, "thresh_beta": None})

    # 7. BorderlineSMOTE
    configs.append({"name": "BorderlineSMOTE", "sampler": "BorderlineSMOTE", "model": "RF", "weight": False, "thresh_beta": None})
    configs.append({"name": "BorderlineSMOTE", "sampler": "BorderlineSMOTE", "model": "XGB", "weight": False, "thresh_beta": None})

    # 8. SMOTE + Tomek
    configs.append({"name": "SMOTETomek", "sampler": "SMOTETomek", "model": "RF", "weight": False, "thresh_beta": None})
    configs.append({"name": "SMOTETomek", "sampler": "SMOTETomek", "model": "XGB", "weight": False, "thresh_beta": None})

    # 9. SMOTE + ENN
    configs.append({"name": "SMOTEENN", "sampler": "SMOTEENN", "model": "RF", "weight": False, "thresh_beta": None})
    configs.append({"name": "SMOTEENN", "sampler": "SMOTEENN", "model": "XGB", "weight": False, "thresh_beta": None})

    # 10. BalancedRandomForest
    configs.append({"name": "BalancedRF", "sampler": None, "model": "BalancedRF", "weight": False, "thresh_beta": None})

    # 11. EasyEnsemble
    configs.append({"name": "EasyEnsemble", "sampler": None, "model": "EasyEnsemble", "weight": False, "thresh_beta": None})

    # 12~13. Threshold tuning (WeightOnly + F1/F2 최적화)
    configs.append({"name": "Thresh_F1", "sampler": None, "model": "RF", "weight": True, "thresh_beta": 1.0})
    configs.append({"name": "Thresh_F1", "sampler": None, "model": "XGB", "weight": True, "thresh_beta": 1.0})
    configs.append({"name": "Thresh_F2", "sampler": None, "model": "RF", "weight": True, "thresh_beta": 2.0})
    configs.append({"name": "Thresh_F2", "sampler": None, "model": "XGB", "weight": True, "thresh_beta": 2.0})

    return configs


def build_model(config, n_pos=None, n_neg=None):
    model_type = config["model"]
    use_weight = config.get("weight", False)

    if model_type == "RF":
        return RandomForestClassifier(
            n_estimators=300, max_depth=8,
            class_weight="balanced" if use_weight else None,
            random_state=RANDOM_STATE, n_jobs=1
        )
    elif model_type == "XGB":
        spw = (n_neg / n_pos) if (use_weight and n_pos and n_pos > 0) else 1
        return XGBClassifier(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0, n_jobs=1
        )
    elif model_type == "BalancedRF":
        return BalancedRandomForestClassifier(
            n_estimators=300, max_depth=8,
            random_state=RANDOM_STATE, n_jobs=1
        )
    elif model_type == "EasyEnsemble":
        return EasyEnsembleClassifier(
            n_estimators=20, random_state=RANDOM_STATE, n_jobs=1
        )


def build_sampler(config, n_minority):
    sampler_name = config.get("sampler")
    if sampler_name is None:
        return None

    ratio = config.get("ratio", 0.5)
    cls_map = {
        "SMOTE": SMOTE,
        "ADASYN": ADASYN,
        "BorderlineSMOTE": BorderlineSMOTE,
        "SMOTETomek": SMOTETomek,
        "SMOTEENN": SMOTEENN,
    }
    cls = cls_map[sampler_name]

    k = min(5, n_minority - 1)
    if k < 1:
        return None

    if cls in (SMOTETomek, SMOTEENN):
        return cls(smote=SMOTE(sampling_strategy=ratio, k_neighbors=k, random_state=RANDOM_STATE),
                   random_state=RANDOM_STATE)
    elif cls == ADASYN:
        return cls(sampling_strategy=ratio, n_neighbors=k, random_state=RANDOM_STATE)
    else:
        return cls(sampling_strategy=ratio, k_neighbors=k, random_state=RANDOM_STATE)


def run_single_experiment(X, Y, config, skf):
    """
    StratifiedKFold CV — fold별 평가 후 macro-average (mean ± std).
    각 fold에 양성이 균등 배분되므로 안정적 평가 가능.
    """
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

        # Impute NaN
        imp = SimpleImputer(strategy="median")
        X_train = imp.fit_transform(X_train)
        X_test = imp.transform(X_test)

        # Sampler (train에만 적용)
        sampler = build_sampler(config, n_pos)
        if sampler is not None:
            try:
                X_train_res, Y_train_res = sampler.fit_resample(X_train, Y_train)
            except Exception:
                skipped += 1
                continue
        else:
            X_train_res, Y_train_res = X_train, Y_train

        # Model
        n_pos_res = int((Y_train_res == 1).sum())
        n_neg_res = int((Y_train_res == 0).sum())
        model = build_model(config, n_pos=n_pos_res, n_neg=n_neg_res)
        model.fit(X_train_res, Y_train_res)

        y_prob = model.predict_proba(X_test)[:, 1]

        # Threshold
        thresh_beta = config.get("thresh_beta")
        if thresh_beta is not None:
            y_prob_train = model.predict_proba(X_train)[:, 1]
            threshold = find_best_threshold(Y_train, y_prob_train, beta=thresh_beta)
        else:
            threshold = 0.5

        metrics = evaluate(Y_test, y_prob, threshold=threshold)
        metrics["fold"] = fold_idx
        metrics["threshold"] = threshold
        metrics["n_pos_train"] = n_pos
        metrics["n_pos_test"] = int(Y_test.sum())
        fold_metrics.append(metrics)

    if not fold_metrics:
        return None

    if skipped > 0:
        print(f"    (경고: {skipped}/{N_SPLITS} fold 스킵)")

    df_folds = pd.DataFrame(fold_metrics)
    summary = {}
    for metric in ["auc_roc", "auc_pr", "recall", "precision", "f1", "f2"]:
        vals = df_folds[metric].dropna()
        summary[f"{metric}_mean"] = vals.mean()
        summary[f"{metric}_std"] = vals.std()
    summary["n_valid_folds"] = len(fold_metrics)
    summary["avg_threshold"] = df_folds["threshold"].mean()
    return summary


def main():
    configs = get_experiments()
    all_results = []

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for coin in ["USDC", "DAI"]:
        print(f"\n{'='*60}")
        print(f"  {coin} 불균형 실험 ({len(configs)} configs)")
        print(f"  CV: StratifiedKFold(n_splits={N_SPLITS}, shuffle=True)")
        print(f"{'='*60}")

        X, Y, feat_names = load_data(coin)
        print(f"  데이터: {X.shape}, 디페깅 비율: {Y.mean()*100:.1f}% ({int(Y.sum())}건)")

        # fold별 양성 수 확인
        for fi, (tr, te) in enumerate(skf.split(X, Y)):
            print(f"    fold {fi}: train={len(tr)} (pos={Y[tr].sum():.0f}), test={len(te)} (pos={Y[te].sum():.0f})")
        print()

        for i, config in enumerate(configs):
            label = f"{config['name']}+{config['model']}"
            t0 = time.time()
            print(f"  [{i+1:2d}/{len(configs)}] {label:30s}", end=" ", flush=True)

            result = run_single_experiment(X, Y, config, skf)

            elapsed = time.time() - t0

            if result is None:
                print(f"→ FAILED ({elapsed:.1f}s)")
                continue

            result["coin"] = coin
            result["technique"] = config["name"]
            result["model"] = config["model"]
            all_results.append(result)

            print(f"→ F2={result['f2_mean']:.3f}±{result['f2_std']:.3f}  "
                  f"Recall={result['recall_mean']:.3f}  Prec={result['precision_mean']:.3f}  "
                  f"AUC-PR={result['auc_pr_mean']:.3f}  ({elapsed:.1f}s)")

    # 결과 저장
    df_results = pd.DataFrame(all_results)
    out_path = os.path.join(ML_DIR, "imbalance_results.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(f"전체 결과 저장: {out_path} ({len(df_results)}행)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
