남태령 대첩의 현장 담론:\
저항 주체의 발화 네트워크 분석을 중심으로
===

남태령 대첩의 ‘새로운 투쟁 세대’ 등장은 담론적 변화를 동반하였는가? 2024년 12월 21~22일 남태령 고개에서 벌어진 농민‧시민 트랙터 시위는 전국농민회총연맹 성명서가 기록하듯 “28 시간 차벽에 막힌 끝에 관저 진격을 성사”시키며 윤석열 대통령 탄핵 국면의 분수령이 되었다(전농 2025). 이미 12월 12일 시민사회가 ‘윤석열 즉각 퇴진·사회대개혁 비상행동’을 발족하며 계엄 시도를 “주권자에 대한 내란”으로 규정했고(비상행동 2024a), 국회 탄핵소추안 가결 직후에는 “주권자 승리”를 선언하는 성명이 발표되었다(비상행동 2024b). 연합뉴스와 뉴시스는 남태령 차벽 앞에서 “윤석열은 방 빼고 경찰은 차 빼라” 구호와 K-팝 응원봉 물결이 밤새 이어졌음을 보도했고, The Guardian 은 이를 “K-팝 문화가 결합된 한국 시위의 진화”로 국제사회에 전했다(Yonhap 2024; Newsis 2024; Guardian 2024). 사건 기록을 집약해둔 캠페인즈닷도의 태그 페이지는 현장 영상과 언론기사 300여 건을 한눈에 제공한다(campaigns.do).

이 연구는 전농 유튜브 스트림 40 시간 분량(음성 236 만 어절)을 클로바노트로 전사·정제해 약 2만 5천 문장 코퍼스를 구축하였다. 텍스트를 Ko-SBERT로 임베딩한 뒤 UMAP(n\_neighbors 15, min\_dist 0.1)으로 2-D 투영하고, HDBSCAN(min\_cluster\_size 35, min\_samples 10)으로 밀도기반 클러스터를 산출하였다. BERTopic을 통해 45개 토픽을 추출했으며, 실루엣 계수 0.5266은 군집 간 분리가 뚜렷함을, 코히런스 0.3587은 발화체 한국어 데이터치고 무난한 의미 응집도를 시사한다. doc\_clarity.csv에서 계산한 명료도는 평균 0.285·중앙값 0.065·상위 10% 1.00의 롱테일 분포였다. 토픽 간 관계는 단어 공출현 행렬을 PPMI로 정규화한 gexf 그래프로 표현했는데, 노드 200·엣지 2 012·밀도 0.101의 희소 네트워크에서 ‘탄핵’·‘민주주의’ 키워드가 평균중심성 0.62로 허브 역할을 하였다.

추출된 상위 토픽은 다섯 축으로 정리된다. 첫째, “농민‧생존권” 토픽(문장 622개)은 ‘농민·농사·전봉준투쟁단’ 등 키워드가 지배해 전통적 농민운동 담론을 재소환했다. 둘째, “경찰 대치” 토픽(419개)은 ‘경찰·차벽·연행’ 어휘로 구성돼 공권력 충돌 양상을 부각했다. 셋째, “창의적 구호” 토픽(402개)은 ‘차 빼라·길 열어’와 같은 현장형 문장을 묶어 시위대의 유머·패러디 전략을 보여 준다. 넷째, “탄핵·헌정수호” 토픽(341개)은 ‘탄핵·민주주의·헌법’ 어휘로 집회 목적을 정당화했으며, 다섯째, “문화제·응원봉” 토픽(274개)은 ‘노래·응원봉·공연’으로 축제적 분위기를 드러냈다. 허브–브리지 분석 결과, 탄핵 토픽이 농민·경찰·문화제 토픽 모두와 평균 PPMI 0.43 이상의 강결 연결을 유지해 메타프레임 구실을 했고, ‘차 빼라’ 토픽은 경찰 토픽과 시민 문화 토픽 사이를 잇는 경계 다리로 나타났다(PPMI 0.31).

이론적 조망으로서, Snow & Benford의 프레이밍 이론은 남태령에서 ‘계엄-질서’ 프레임을 ‘민주주의-생존권’ 프레임이 압도하는 과정을 해석하는 데 유용하고(Snow & Benford 1992), Fligstein & McAdam의 전략적 행동장 모델은 정부·경찰과 농민·시민의 장(場) 경쟁을 구조적으로 조망해 준다(Fligstein & McAdam 2012). Melucci가 말한 집합정체성 형성은 조직 없는 현장 연대가 어떻게 즉흥적으로 공동의 ‘우리’를 구성했는지를 설명한다(Melucci 1989).

방법론적으로 본 연구는 대용량 구호‧연설체를 자동 전사·정제해 토픽 모델링과 시멘틱 네트워크 분석을 결합한 사례다. HDBSCAN의 파라미터를 달리해 본 결과 min\_cluster\_size·min\_samples가 15 % 이상 작아지면 실루엣이 0.42 이하로 급락했고, UMAP min\_dist를 0.05로 줄이면 토픽 간 중첩이 늘어나 코히런스가 0.33까지 떨어졌다. 따라서 현 파라미터 세트는 군집 응집도와 토픽 해석 가능성 간 균형점으로 판단된다. 또한 PPMI 네트워크의 모듈러리티는 0.29로 나타나 담론 하위공동체가 느슨하게 분리된 구조임을 뒷받침한다.

본 연구의 한계도 분명하다. 첫째, ASR 전사 특성상 화자 인식이 정확하지 않아 발화 주체별 프레임 차이를 정밀하게 비교하지 못했다. 둘째, 현장 텍스트만 다루었기에 SNS·언론 2차 담론과의 상호작용은 분석하지 않았다. 셋째, 자동 번역·외래어 정제 과정에서 의미 손실 위험이 남아 있다. 그럼에도 본 연구는 계엄 국면이라는 고위험 상황에서 농민과 도시 시민이 디지털 네트워크 기반으로 의미를 조정·확산하는 메커니즘을 실증적으로 제시했다는 점, 그리고 사회운동 연구에 최신 NLP 도구(BERTopic·UMAP·HDBSCAN·PPMI)를 통합한 재현가능 파이프라인을 제공했다는 점에서 기여가 있다. 결론적으로 남태령 대첩의 새로운 투쟁 세대는 다층적 담론을 구축하며 농민·시민·청년을 엮는 의미 연대를 형성하였고, 이는 2016 촛불 이후 한국 사회운동이 프레임·조직·레퍼토리 측면에서 한 걸음 진화했음을 보여준다.

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
├─ experiments.py          # 다양한 수치 실험
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
!python run_pipeline.py --data data/n1.csv data/n2.csv
```

### 혹은 다양한 값을 넣어 실험
```bash
!python experiments.py --data data/n1.csv data/n2.csv
```

***

## 연구자: 정윤석 (고려대학교)

* 문의: soci.yunseok.jeong [at] gmail [dot] com
* 혹은: jeongyunseok [at] jinbo [dot] net
