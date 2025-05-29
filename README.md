# 한국사회학회 전기학술대회 남태령 대첩 발표자료용
1차 및 2차 남태령 대첩 시민발언 스크립트를 기반으로\
임베딩 → 차원 축소 → HDBSCAN 클러스터링 → BERTopic 토픽 모델링 → PPMI 네트워크 구축까지\
원클릭으로 실행 가능한 파이프라인입니다.

***

## 주요 기능

* **한국어 토큰화**\
  KiwiPiePy 기반으로 NNG/NNP/SL 품사만 추출

* **문장 임베딩**\
  `beomi/kcbert-large` 모델을 활용한 mean-pool 방식

* **차원 축소**\
  UMAP (기본 3차원)

* **클러스터링**\
  HDBSCAN 그리드 탐색 → 최적 silhouette score 선택

* **토픽 모델링**\
  BERTopic (`nr_topics` 고정 또는 auto)

* **PPMI 네트워크**\
  상위 N개 단어 간 PPMI 계산 → 상위 P% 엣지만 보존

* **자동화된 결과 출력**

  * `pipeline_output.csv` — 각 문장별 토픽/클러스터 할당

  * `exports/topic_summary.csv` — 토픽별 대표 단어 및 예시

  * `exports/graph_ppmi.gexf` — Gephi/Flourish 로 읽을 수 있는 네트워크

***

## 저장소 구조

```
ksa_25_test_1/
├─ data/                   # 원본 CSV (n1.csv, n2.csv)
├─ models/                 # HF 모델·임베딩 캐시
├─ exports/                # 결과물 (CSV, GEXF)
├─ config.py               # 파라미터·경로 설정
├─ data_load.py            # 데이터 로드 유틸
├─ preprocess.py           # 토큰화·전처리
├─ tokenizer.py            # 토큰화 래퍼
├─ embedding.py            # KC-BERT 임베딩
├─ cluster.py              # UMAP + HDBSCAN
├─ topic_model.py          # BERTopic 래핑
├─ graph_build.py          # PPMI 네트워크 생성
├─ metrics.py              # 군집·그래프 메트릭 계산
├─ visualize.py            # CSV/GEXF 내보내기
├─ pipeline.py             # 전체 파이프라인 조합
└─ run_pipeline.py         # 엔트리포인트(자동 파일 검색)
```

***

## 설치 및 실행

* Colab A100 기준입니다. CPU 사용시 느릴 수 있습니다.

### 1. 

```bash
# ---------------------- SUPER-LIGHT SETUP CELL (final) ----------------------
import os, subprocess, sys, importlib

REPO_URL = "https://github.com/socijeongyunseok/ksa_25_test_1.git"
REPO_DIR = "ksa_25_test_1"

# 1) Git clone / pull
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--quiet", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--quiet"], check=True)

# 2) pip install (필요한 것만)
REQS = [
    "pandas", "bertopic", "umap-learn", "hdbscan",
    "kiwipiepy", "mecab-python3",         # 형태소기
    "unidic-lite",                        # 경량 사전
    "networkx", "gensim"
]
missing = [pkg for pkg in REQS if importlib.util.find_spec(pkg) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

# 3) 이동 & autoreload
os.chdir(REPO_DIR)

import IPython
ip = IPython.get_ipython()
if ip:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("✅  Setup complete.  Repository =", REPO_DIR)```
```

### 2. 파이프라인 실행

```bash
!python run_pipeline.py n1.csv n2.csv
```

### 혹은 다양한 값을 넣어 실험
```bash
!python experiments.py \
  --data data/n1.csv data/n2.csv \
  --grid_results exports/grid_results.csv
```

***

## 라이선스 & 문의

* 문의: soci.yunseok.jeong [at] gmail [dot] com
* 혹은: jeongyunseok [at] jinbo [dot] net
