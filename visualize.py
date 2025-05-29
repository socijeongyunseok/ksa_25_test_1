"""
visualize.py
-------------
UMAP 2D 산점도 & 실루엣 분포 플롯
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples


def plot_umap(umap_embeddings: np.ndarray, topics: np.ndarray, probs: np.ndarray):
    """
    투명도 = p(top1). 색 = hard label
    """
    import seaborn as sns  # 시각화 가독성용 (Colab 기본 설치)
    sns.set(style="whitegrid")

    top1_prob = probs.max(axis=1)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=topics,
        s=50,
        alpha=top1_prob,   # 명확도 높은 문서 = 불투명
        cmap="tab20",
    )
    plt.colorbar(scatter, label="Topic ID")
    plt.title("UMAP projection of topics (alpha = top1 prob)")
    plt.tight_layout()
    plt.show()


def plot_silhouette(embeddings: np.ndarray, labels: np.ndarray):
    """
    클러스터별 실루엣 스코어 분포
    """
    sil = silhouette_samples(embeddings, labels)
    plt.figure(figsize=(8, 4))
    plt.hist(sil, bins=40, alpha=0.8)
    plt.xlabel("Silhouette score")
    plt.ylabel("Documents")
    plt.title(f"Silhouette distribution (mean={sil.mean():.3f})")
    plt.tight_layout()
    plt.show()
