"""
Colab용: FinBERT 임베딩 → 디페깅 프로토타입 유사도 계산

사용법:
1. Google Colab에서 이 파일 실행
2. news_raw.csv를 Colab에 업로드
3. 결과 embedding_features.csv를 다운로드 → data/processed/에 저장

GPU 런타임 권장 (T4 기준 ~10분)
"""

# ── 설치 ──
# !pip install transformers torch -q

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import files
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# 0. 설정
# ═══════════════════════════════════════════════════════════════

MODEL_NAME = "yiyanghkust/finbert-tone"
BATCH_SIZE = 64
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# USDC + DAI 합집합 디페깅 날짜 (TP + ±1% + 2-of-3 기준)
DEPEG_DATES = [
    '2020-01-28', '2020-01-29', '2020-01-30', '2020-01-31',
    '2020-02-01', '2020-02-02', '2020-02-03', '2020-02-04',
    '2020-02-05', '2020-02-06', '2020-02-07', '2020-02-08',
    '2020-02-09', '2020-02-10', '2020-02-11', '2020-02-12',
    '2020-02-13', '2020-02-14', '2020-02-15', '2020-02-16',
    '2020-02-17', '2020-02-18', '2020-02-19', '2020-02-20',
    '2020-02-21', '2020-03-08', '2020-03-13', '2020-03-14',
    '2020-03-15', '2020-03-16', '2020-03-17', '2020-03-18',
    '2020-03-19', '2020-03-20', '2020-03-21', '2020-03-23',
    '2020-03-24', '2020-03-25', '2020-03-26', '2020-04-01',
    '2020-04-02', '2020-04-03', '2020-04-04', '2020-04-05',
    '2020-04-06', '2020-04-07', '2020-04-08', '2020-04-09',
    '2020-04-10', '2020-04-17', '2020-04-18', '2020-04-19',
    '2020-04-20', '2020-04-24', '2020-04-25', '2020-04-26',
    '2020-04-27', '2020-05-14', '2020-05-15', '2020-07-04',
    '2020-07-05', '2020-07-06', '2020-07-08', '2020-07-09',
    '2020-07-10', '2020-07-11', '2020-07-12', '2020-07-13',
    '2020-07-14', '2020-07-22', '2020-07-23', '2020-07-24',
    '2020-07-25', '2020-07-26', '2020-07-27', '2020-07-28',
    '2020-07-29', '2020-07-30', '2020-07-31', '2020-08-01',
    '2020-08-02', '2020-08-03', '2020-08-04', '2020-08-05',
    '2020-08-06', '2020-08-07', '2020-08-08', '2020-08-09',
    '2020-08-10', '2020-08-11', '2020-08-12', '2020-08-13',
    '2020-08-14', '2020-08-15', '2020-08-25', '2020-08-26',
    '2020-08-27', '2020-08-28', '2020-08-29', '2020-08-30',
    '2020-08-31', '2020-09-01', '2020-09-02', '2020-09-03',
    '2020-09-04', '2020-09-05', '2020-09-06', '2020-09-07',
    '2020-09-08', '2020-09-09', '2020-09-10', '2020-09-11',
    '2020-09-12', '2020-09-13', '2020-09-14', '2020-09-15',
    '2020-09-16', '2020-09-17', '2020-09-18', '2020-09-19',
    '2020-09-20', '2020-09-21', '2020-09-22', '2020-09-23',
    '2020-09-24', '2020-09-25', '2020-09-26', '2020-09-27',
    '2020-09-28', '2020-09-29', '2020-09-30', '2020-10-02',
    '2020-10-04', '2020-10-05', '2020-10-06', '2020-10-07',
    '2020-10-08', '2020-10-09', '2020-10-10', '2020-10-11',
    '2020-10-12', '2021-11-17', '2021-11-18', '2023-03-12',
    '2023-03-13',
]

# ═══════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════════════════════════════

print("news_raw.csv 업로드해주세요...")
uploaded = files.upload()
filename = list(uploaded.keys())[0]

df = pd.read_csv(filename)
df["Date"] = pd.to_datetime(df["Date"])
df["text_input"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str[:512]

print(f"로드 완료: {df.shape[0]}건, {df['Date'].min().date()} ~ {df['Date'].max().date()}")
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# 2. FinBERT 임베딩 생성
# ═══════════════════════════════════════════════════════════════

print(f"\nFinBERT 로딩... ({MODEL_NAME})")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def get_embeddings(texts, batch_size=BATCH_SIZE):
    """배치 단위로 [CLS] 토큰 임베딩 추출"""
    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(cls_embeddings.cpu().numpy())

        batch_num = i // batch_size + 1
        if batch_num % 20 == 0 or batch_num == n_batches:
            print(f"  batch {batch_num}/{n_batches}")

    return np.vstack(all_embeddings)


print(f"임베딩 생성 중... ({len(df)}건, batch_size={BATCH_SIZE})")
embeddings = get_embeddings(df["text_input"].tolist())
print(f"임베딩 완료: {embeddings.shape}")

# ═══════════════════════════════════════════════════════════════
# 3. 디페깅 프로토타입 생성
# ═══════════════════════════════════════════════════════════════

depeg_dates_set = set(pd.to_datetime(DEPEG_DATES))
is_depeg = df["Date"].isin(depeg_dates_set)

depeg_embeddings = embeddings[is_depeg.values]
normal_embeddings = embeddings[~is_depeg.values]

print(f"\n디페깅 기사: {depeg_embeddings.shape[0]}건")
print(f"정상 기사: {normal_embeddings.shape[0]}건")

# 프로토타입 = 평균 임베딩
depeg_prototype = depeg_embeddings.mean(axis=0).reshape(1, -1)
normal_prototype = normal_embeddings.mean(axis=0).reshape(1, -1)

# ═══════════════════════════════════════════════════════════════
# 4. 기사별 유사도 계산
# ═══════════════════════════════════════════════════════════════

print("유사도 계산 중...")
sim_depeg = cosine_similarity(embeddings, depeg_prototype).flatten()
sim_normal = cosine_similarity(embeddings, normal_prototype).flatten()

df["sim_depeg"] = sim_depeg
df["sim_normal"] = sim_normal
df["sim_gap"] = sim_depeg - sim_normal  # 양수면 디페깅 뉴스에 더 가까움

# ═══════════════════════════════════════════════════════════════
# 5. 일별 집계
# ═══════════════════════════════════════════════════════════════

daily = df.groupby("Date").agg(
    emb_sim_depeg_mean=("sim_depeg", "mean"),
    emb_sim_depeg_max=("sim_depeg", "max"),
    emb_sim_normal_mean=("sim_normal", "mean"),
    emb_sim_gap_mean=("sim_gap", "mean"),
    emb_sim_gap_max=("sim_gap", "max"),
    emb_news_count=("sim_depeg", "count"),
).reset_index()

# 파생 변수
for lag in [1, 3, 7]:
    daily[f"emb_sim_depeg_mean_lag{lag}"] = daily["emb_sim_depeg_mean"].shift(lag)
    daily[f"emb_sim_gap_mean_lag{lag}"] = daily["emb_sim_gap_mean"].shift(lag)

daily["emb_sim_depeg_ma7"] = daily["emb_sim_depeg_mean"].rolling(7).mean()
daily["emb_sim_depeg_spike"] = (
    daily["emb_sim_depeg_mean"] > daily["emb_sim_depeg_ma7"] * 1.5
).astype(int)

daily["emb_sim_gap_ma7"] = daily["emb_sim_gap_mean"].rolling(7).mean()

print(f"\n일별 집계 완료: {daily.shape[0]}일, {daily.shape[1] - 1}개 변수")

# ═══════════════════════════════════════════════════════════════
# 6. 검증: 디페깅 시기 유사도 확인
# ═══════════════════════════════════════════════════════════════

print("\n디페깅 주요 시기 유사도:")
for period, start, end in [
    ("코로나 (2020-03)", "2020-03-01", "2020-03-31"),
    ("DeFi Summer (2020-08)", "2020-08-01", "2020-08-31"),
    ("SVB (2023-03)", "2023-03-01", "2023-03-31"),
    ("정상기 (2024-06)", "2024-06-01", "2024-06-30"),
]:
    sub = daily[(daily["Date"] >= start) & (daily["Date"] <= end)]
    if len(sub) > 0:
        print(f"  {period}: sim_depeg={sub['emb_sim_depeg_mean'].mean():.4f}, "
              f"sim_gap={sub['emb_sim_gap_mean'].mean():.4f}")

# ═══════════════════════════════════════════════════════════════
# 7. 저장 + 다운로드
# ═══════════════════════════════════════════════════════════════

out_filename = "embedding_features.csv"
daily.to_csv(out_filename, index=False)
print(f"\n저장 완료: {out_filename}")

files.download(out_filename)
print("다운로드 시작! → data/processed/ 폴더에 넣어주세요")
