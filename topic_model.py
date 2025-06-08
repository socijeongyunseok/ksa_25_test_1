# -*- coding: utf-8 -*-

# ============================================================
# topic_model.py
# ------------------------------------------------------------
from __future__ import annotations
from typing import Tuple, List
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer

# ── 1. 클러스터링/벡터라이저 하이퍼파라미터 ───────────────────
HDBSCAN_KWARGS = dict(
    min_cluster_size=35,
    min_samples=10,
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
VEC_KWARGS = dict(
    min_df=3,
    max_df=0.8,
    ngram_range=(1, 3),
    stop_words=[
        "그리고","그러나","그러므로",
        "안녕","안녕하세요","안녕하십니까",
        "인사","인사드리겠습니다","인사드립니다",
        "투쟁으로","차빼라","열어라",
        "합니다","입니다","니까","이","저","그",
        "감사","고맙"
    ],
)

# ── 2. 기본 토픽 모델 ────────────────────────────────────────────

def build_topic_model(texts: List[str]) -> Tuple[BERTopic, List[int], "np.ndarray"]:
    """텍스트 리스트 → (BERTopic 모델, 문서당 토픽, 문서당 확률)"""
    umap_model = UMAP(**UMAP_KWARGS)
    hdbscan_model = hdbscan.HDBSCAN(**HDBSCAN_KWARGS)
    vectorizer = CountVectorizer(**VEC_KWARGS)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=True,
        language="multilingual",
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(texts)

    # ── 작은 토픽 병합 ──────────────────────────────────────────
    SMALL_SIZE = 10
    info = topic_model.get_topic_info()
    small_topics = info[info["Count"] < SMALL_SIZE].Topic.tolist()
    if small_topics:
        result = topic_model.merge_topics(texts, small_topics)
        if isinstance(result, tuple) and len(result) == 3:       # 0.16+
            topic_model, topics, probs = result
        elif result is not None:
            topics, probs = result
    return topic_model, topics, probs

# ── 3. 토픽-오버-타임 래퍼 ────────────────────────────────────────

def build_topic_timeseries(texts: List[str],
                           times: List[int],
                           nr_bins: int = 10) -> Tuple[BERTopic, pd.DataFrame]:
    """토픽 모델 + 시계열 DF 반환"""
    topic_model, _, _ = build_topic_model(texts)
    tot_df = topic_model.topics_over_time(docs=texts,
                                          timestamps=times,
                                          nr_bins=nr_bins,
                                          evolution_tuning=True)
    return topic_model, tot_df

