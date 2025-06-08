# -*- coding: utf-8 -*-

# run_pipeline.py
# ------------------------------------------------------------
import pprint
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from config import DATA_DIR
from data_load import load_dialog_csv
from topic_model import build_topic_timeseries
from experiments import compute_silhouette  # 기존 기능 유지

# ── CSV 자동 검색 ───────────────────────────────────────────────

def find_csvs():
    p = Path(DATA_DIR)
    csvs = sorted(p.glob("n*.csv"))
    if len(csvs) < 2:
        raise FileNotFoundError(f"{DATA_DIR} 폴더에 n1.csv, n2.csv 두 파일이 필요합니다.")
    return [c.name for c in csvs[:2]]  # 첫 두 개 사용

# ── 단일 CSV 실행 헬퍼 ─────────────────────────────────────────

def run_single(csv_name: str, nr_bins: int):
    df = load_dialog_csv(csv_name)
    model, tot = build_topic_timeseries(df["text"].tolist(),
                                        df["timestamp"].tolist(),
                                        nr_bins=nr_bins)
    # silhouette 계산
    embeds = model._embedding_model.encode(df["text"].tolist(), show_progress_bar=False)
    labels = model.get_document_info(df["text"].tolist())['Topic'].to_numpy()
    try:
        silh = compute_silhouette(embeds, labels)
    except Exception:
        silh = float("nan")
    metrics = {"silhouette": silh, "n_topics": len(set(labels))-(-1 in labels)}
    tot["source"] = csv_name
    return tot, metrics, model

# ── GEXF 네트워크 빌더 ─────────────────────────────────────────

def build_gexf(topic_model: "BERTopic", export_dir: Path,
               thresh: float = 0.40, k: int = 4,
               prefix: str = "graph"):
    """토픽 임베딩 코사인 유사도로 GEXF 저장"""
    emb = np.asarray(topic_model.topic_embeddings_)
    if emb.ndim != 2:
        print("⚠️  topic_embeddings_가 없습니다. 스킵합니다.")
        return None

    sim = cosine_similarity(emb)

    # ID → label 매핑 (번호: 대표 단어 5개)
    topic_info = topic_model.get_topic_info()
    id_to_label = topic_info.set_index("Topic")["Name"].to_dict()

    edges = []
    for i in range(len(id_to_label)):
        sims = [(j, sim[i, j]) for j in range(len(id_to_label)) if j != i and sim[i, j] >= thresh]
        sims = sorted(sims, key=lambda x: x[1], reverse=True)[:k]
        for j, w in sims:
            edges.append((id_to_label.get(i, str(i)), id_to_label.get(j, str(j)), float(w)))

    if not edges:
        print("⚠️  조건을 만족하는 엣지가 없습니다. THRESH/K 값을 조정해 보세요.")
        return None

    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    nx.set_node_attributes(G, {n: n for n in G.nodes()}, "label")

    out_path = export_dir / f"{prefix}.gexf"
    nx.write_gexf(G, out_path)
    print(f"✓ 토픽 네트워크 GEXF로 저장됨: {out_path}")
    return out_path

# ── 메인 ──────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nr_bins", type=int, default=10, help="시계열 bin 개수")
    p.add_argument("--export_dir", default="exports", help="출력 폴더")
    p.add_argument("--thresh", type=float, default=0.40, help="코사인 유사도 최소값")
    p.add_argument("--topk", type=int, default=4, help="노드당 최대 엣지")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    export_dir = Path(args.export_dir)
    export_dir.mkdir(exist_ok=True)

    csv_list = find_csvs()
    all_tot = []
    metrics_dict = {}

    # 파일별 모델 저장: {csv: model}
    model_dict = {}

    for csv_name in csv_list:
        print(f"▶ 분석 시작: data/{csv_name}")
        tot, m, model = run_single(csv_name, nr_bins=args.nr_bins)
        all_tot.append(tot)
        metrics_dict[csv_name] = m
        model_dict[csv_name] = model

    # TOT 결과 합치기 & 저장
    df_tot = pd.concat(all_tot, ignore_index=True)
    tot_out = export_dir / "topic_over_time.csv"
    df_tot.to_csv(tot_out, index=False, encoding="utf-8-sig")

    # 각 파일·전체 GEXF 생성
    for csv_name, model in model_dict.items():
        build_gexf(model, export_dir, thresh=args.thresh, k=args.topk,
                   prefix=Path(csv_name).stem)

    # 글로벌 모델(두 파일 합친 텍스트)로도 네트워크 만들기
    global_texts = load_dialog_csv(*csv_list)["text"].tolist()
    global_model, _ = build_topic_timeseries(global_texts,
                                             list(range(len(global_texts))),
                                             nr_bins=args.nr_bins)
    build_gexf(global_model, export_dir, thresh=args.thresh, k=args.topk,
               prefix="global")

    pprint.pprint(metrics_dict, compact=True)
    print(f"
✓ 파이프라인 완료 — TOT: {tot_out} 
  GEXF → {export_dir}/*.gexf 확인")
