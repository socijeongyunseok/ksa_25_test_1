# -*- coding: utf-8 -*-

from pathlib import Path

# === 경로 설정 ===
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / 'data'
MODEL_DIR   = BASE_DIR / 'models'
EXPORT_DIR  = BASE_DIR / 'exports'

# === 랜덤 시드 ===
RANDOM_SEED = 42

# === UMAP 하이퍼파라미터 ===
UMAP_N_COMPONENT       = 3                  # 3D 공간에서 더 뚜렷한 군집 분리
UMAP_MIN_DIST_LIST     = [0.0, 0.02, 0.05]  # 0.0 과 0.02 조합으로 실루엣 0.2↑ 목표
UMAP_N_NEIGHBORS_LIST  = [5, 10, 15]        # 지역 구조 세밀 탐색

# === HDBSCAN 하이퍼파라미터 ===
HDBSCAN_MIN_CLUS            = (15, 20)      # 너무 작은 군집 제외, 의미 있는 덩어리 형성
HDBSCAN_MIN_SAMP            = (5, 10)       # 노이즈 균형 조절
CLUSTER_SELECTION_EPSILON   = [0.0, 0.02]   # 약간의 경계 완충으로 경계 선명도 제어

# === 문서 필터링 ===
MAX_TOKEN_REPEAT       = 10  # 동일 단어 과도 반복 문서 제거
MIN_TOKEN_COUNT        = 5   # 너무 짧은 문장 제거

# === 토픽 모델링 ===
TOPIC_TOP_N_WORD       = 15
NR_TOPICS              = 15
TOPIC_CANDIDATES       = [15, 20]          # 핵심 후보 두 개로 자동 선택
NGRAM_RANGE            = (1, 3)            # 트라이그램까지 포함
MIN_TOPIC_SIZE         = 20                # 작으면 자동 병합

# === 네트워크 생성 ===
N_TOP_WORDS            = 200
# PPMI: 핵심 공기 관계(상위 25%) 유지
# Jaccard: 핵심 유사도(임계값 0.12) 유지
GRAPH_METRIC           = 'ppmi'  
PMI_TOP_P              = 0.25
JACCARD_THRESH         = 0.12
