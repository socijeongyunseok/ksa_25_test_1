"""
topic_model.py
---------------
BERTopic 생성 & 학습 함수
  * 파라미터 튜닝 완판
"""

from __future__ import annotations
from typing import Tuple, List
from bertopic import BERTopic
from umap import UMAP
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer


# ----------------------------------------------------------------------
# 1) UMAP & HDBSCAN 파라미터 튜닝
HDBSCAN_KWARGS = dict(
    min_cluster_size=35,     # 20 → 35
    min_samples=10,          # 그대로
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

UMAP_KWARGS = dict(
    n_neighbors=15,
    n_components=5,
    min_dist=0.10,
    metric="cosine",
    random_state=42,
)

# ----------------------------------------------------------------------
# 2) CountVectorizer 파라미터 (VEC_KWARGS)
VEC_KWARGS = dict(
    min_df=3,                # 5 → 3
    max_df=0.8,              # 새로 추가
    ngram_range=(1, 3),      # (1,1) → (1,3)
    stop_words=[
        "그리고","그러나","그러므로",
        "안녕","안녕하세요","안녕하십니까",
        "인사","인사드리겠습니다","인사드립니다",
        "투쟁으로","차빼라","열어라",
        "합니다","입니다","니까","이","저","그",
        "감사","고맙"
    ],
)


MIN_TOPIC_SIZE = 30
TOPIC_DIVERSITY = 0.30

# ----------------------------------------------------------------------
# 2) build_topic_model
# ----------------------------------------------------------------------
def build_topic_model(
    texts: List[str],
) -> Tuple[BERTopic, List[int], "np.ndarray"]:
    """
    texts -> (topic_model, topics, probs)
    """
    umap_model = UMAP(**UMAP_KWARGS)
    hdbscan_model = hdbscan.HDBSCAN(**HDBSCAN_KWARGS)
    vectorizer = CountVectorizer(**VEC_KWARGS)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=True,
        language="multilingual",       # 한·영 혼용이면
        verbose=False,
    )

    topics, probs = topic_model.fit_transform(texts)

    # ── 작은 토픽 병합 ─────────────────────────────────────────
    MIN_TOPIC_SIZE = 10
    info = topic_model.get_topic_info()
    small_topics = info[info["Count"] < MIN_TOPIC_SIZE].Topic.tolist()
    print("▶ small_topics to merge:", small_topics)

    if small_topics:
        result = topic_model.merge_topics(texts, small_topics)

        if result is None:                                 # ≤0.15
            topics, probs = topic_model.transform(texts)
        elif isinstance(result, tuple):
            if len(result) == 3:                           # ≥0.16
                topic_model, topics, probs = result
            else:                                          # 0.15.1〜0.15.5
                topics, probs = result

        # BERTopic 0.16+ → (new_model, topics, probs)
        if isinstance(result, tuple) and len(result) == 3:
            topic_model, topics, probs = result
        else:  # 혹시 구버전이면 in-place 수정 → result = (topics, probs)
            topics, probs = result
    return topic_model, topics, probs
