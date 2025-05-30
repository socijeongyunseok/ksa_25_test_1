남태령 대첩의 현장 담론:\
저항 주체의 발화 네트워크 분석을 중심으로
===

‘남태령 담론’이란 무엇인가? 2024년 12월 21-22일 남태령에서 벌어진 전국농민회총연맹(전농)의 트랙터 시위는 윤석열 계엄에 맞선 사회대전환 투쟁의 결정적 장면이었다(Park 2025). ‘양곡관리법’ 거부를 시작으로 전레 없는 거부권 정치를 자행해온 윤석열의 계엄 내란과 뒤이은 탄핵 소추안 가결이라는 정치적 격동 속에서 전농 농민들은 윤석열 체포와 탄핵 인용을 촉구하며 대규모 트랙터 시위를 조직하였다. 약 30여 대의 트랙터와 50여 대의 화물차로 이루어진 농민 시위대는 서울 도심으로 향했으나, 남태령 고개에서 경찰 차벽에 가로막혀 진입이 좌절되었다. 남태령에서의 농민과 경찰의 대치는 무려 28시간 넘게 지속되었고, 그 사이 해당 소식이 트위터 등 소셜 미디어를 통해 퍼지자 탄핵 촛불 집회에 참여하던 ‘응원봉 시민’ 수천 명이 남태령으로 몰려와 농민들을 연대 지지하였다. 밤새 시민들은 현장에서 농민들과 함께 구호를 외치고 물품을 지원했으며, 경찰의 강제 해산 시도에 “윤석열은 방 빼고, 경찰은 차 빼라” 등의 창의적인 구호로 맞섰다. 결국 12월 22일 오후 경찰이 길을 열고 일부 트랙터의 서울 진입을 허용함으로써 남태령 대치는 농민·시민의 사실상 승리로 막을 내렸다. 전농은 사후 성명서에서 “부동산·농업 정책 실패와 위헌적 계엄 쿠데타가 겹친 국가 위기를 농민과 시민이 함께 돌파한 사건”이라 명명했다.(전농 2025).

남태령 대첩은 전통적인 농민운동에 도시 청년 세대가 대거 합류한 새로운 연대의 장이었다는 점에서 사회운동사적으로 중요한 의미를 지닌다. 실제로 2025년 3월 26-27일 진행된 제2차 남태령 투쟁에서 제1차 남태령 투쟁의 기억은 여러 차례 ‘기적’이라 지칭된다. 따라서 본 연구는 남태령 이후 운동 담론의 원형이 되는 1·2차 남태령 대첩의 현장 발화를 분석함으로써 남태령 대첩을 통한 ‘새로운 투쟁 주체 형성’이 어떠한 담론적 변화를 창출했는지를 밝히고자 한다.

이에 본 연구는 1·2차 남태령 대첩의 발화를 실시간 중계한 전농 유튜브 스트림 약 36시간 37분 분량(음성 236 만 어절)을 클로바노트 API로 전사·정제해 약 2만 5천 문장 코퍼스를 구축하였다. 방법론적으로 본 연구는 대용량 구호‧연설체를 자동 전사·정제해 토픽 모델링과 시멘틱 네트워크 분석을 결합했다. 텍스트를 Ko-SBERT로 임베딩한 뒤 UMAP(n_neighbors 15, min_dist 0.1)으로 2-D 투영하고, HDBSCAN(min_cluster_size 35, min_samples 10)으로 밀도기반 클러스터를 산출하였다. 이를 통해 본 연구는 BERTopic을 통해 45개 토픽을 추출했으며, 200개 노드와 2012 에지를 갖는 연결망을 형성했다. 실루엣 계수 0.5266은 군집 간 분리가 뚜렷함을, 코히런스 0.3587은 비정제 발화체 한국어 데이터로서 무난한 의미 응집도를 시사한다.

추출된 상위 토픽은 여섯 축으로 정리된다. 먼저 블록 A-D는 '남태령 대첩'의 핵심 담론을 구성한다. 블록 A (농민 생존·농업 위기)에서, 농민 단체는 진단 프레임을 통해 가격 폭락과 정부 무책임을 ‘불의’로 명명하며 책임소재를 분명히 한다. 동시에 ‘응원봉 동지’로 불리는 탄핵 집회 참여자들과의 프레임 브리징을 통해 주변적 도전자에서 전략적 행위 장의 브로커로 부상한다. 이러한 브리징은 “프레임 정합”이 동원 저변을 확장한다는 주장과 들어맞는다. 블록 B (국가 폭력·경찰 대치)에서, “차빼라” 구호는 현장 억압을 실시간으로 문제화하며 집단 정서를 고조시키는 동기 프레임이다. 이 전술 혁신은 환경 판독과 즉각적 의미 부여 능력 등 시민 진영이 지닌 운동 역량의 산물로, 경찰과의 힘겨루기 속에서도 참여자를 결집시킨다. 블록 C (헌정질서·민주주의 위기) 에서, “모든 권력은 시민에게”라는 헌법 인용은 농민 의제와 응원봉 시민 의제를 포괄하는 마스터 프레임으로 기능한다. 쿠데타 규탄이라는 진단과 민주 질서 복원이라는 처방을 동시에 수행함으로써, 국가·사법 장과 수직적으로 연결되어 투쟁의 제도적 정당성을 강화한다. 마지막으로 블록 D (퇴진·처벌 직접 요구) 에서, ‘내란수괴 체포’ 같은 급진 슬로건은 동기 프레임을 극대화한다. 다음으로 또한 블록 E-F는 '남태령 대첩'의 주변 담론을 구성한다. 블록 E (연대·문화·의례 자원) 에서, 노래·합창·퍼포먼스 등은 정서적 동기 프레임으로서 기능한다. 문화예술 소집단은 투쟁의 하위장 역할을 맡아, 공포·피로를 상쇄해 동원 지속성을 확보하며, 장기화되는 투쟁을 위한 ‘재충전 회로’를 제공한다. 블록 F (현장 운영·돌봄 인프라) 에서, 현장 운영팀은 '거버넌스 유닛'으로 작동하며, 핫팩·분실물 센터 등 돌봄 활동은 투쟁의 처방 프레임을 물질적으로 구체화한다.

본 연구의 한계도 분명하다. 첫째, 자동 음성 인식 전사 특성상 화자 인식이 정확하지 않아 발화 주체별 프레임 차이를 정밀하게 비교하지 못했다. 둘째, 현장 발언만 다루었기에 SNS·언론 2차 담론과의 상호작용은 분석하지 않았다. 셋째, 자동 전사·외래어 정제 과정에서 의미 손실 위험이 크게 남아 있다. 추후 연구에서는 발화자 태깅을 통해 담론을 발화 주체별로 분류한 뒤, 프레임의 미묘한 차이를 분석하는 기법을 도입할 수 있을 것이다. 나아가 SNS와 언론 보도 등 외부 담론장과의 상호작용을 통합적으로 살펴보는 것도 의미가 있다. 그럼에도 본 연구는 계엄 국면이라는 고위험 상황에서 농민과 도시 시민이 디지털 네트워크 기반으로 의미를 조정·확산하는 메커니즘을 실증적으로 제시했다는 점, 그리고 사회운동 연구에 BERTopic·UMAP·HDBSCAN·PPMI 등 최신 자연어처리 도구를 통합한 재현 가능한 파이프라인을 제공했다는 점에서 기여한 바 있다.

결론적으로 남태령 대첩의 새로운 투쟁 세대는 메타프레임들을 가진 다(多)의제 연합으로서, 다층적 담론을 구축하며 농민·시민·청년을 엮는 의미 연대를 형성하였다. 남태령 대첩의 담론적 의미를 돌아보면, 이 사건은 한국 민주주의 역사에서 집합적 정체성 형성과 민주주의 언어의 재구성이라는 측면에서 중요한 전환점으로 평가된다. 다양한 세대와 계층이 “응원봉”과 “트랙터”라는 상징 아래 모여들어 새로운 ‘시민-농민 연대’의 주체를 형성하였고, 이 과정에서 과거 반독재 민주화 운동부터 현대 K팝 문화까지 아우르는 폭넓은 언어 자원을 동원하여 민주주의 담론을 업데이트했다. 그 결과 “남태령”은 단순한 지명이 아니라 세대교체형 시민운동의 아이콘으로 기억되기에 이르렀다. 이는 2016-17년 촛불 항쟁이후 약 7년만에 나타난 대중동원의 새로운 물결로서, 한국 시민사회가 여전히 살아있고 진화하고 있음을 보여주는 사례다. 또한 남태령 대첩은 농민운동사적으로도 획기적인 의미를 갖는다. 과거 농민운동이 주로 농촌에 국한된 이슈로 다루어졌다면, 이제는 도시 시민들과 어깨를 걸고 공동의 민주주의 의제를 외치는 연대의 새 지평을 열었다는 점에서 2016년 박근혜 탄핵 촛불 이후 남한 사회운동이 농민운동을 매개로 프레임·조직·레퍼토리 측면의 진화를 기대해볼 수 있음을 보여준다. 특히 한국 사회가 직면한 세대 갈등, 도농 간격 등을 넘어서는 범사회적 연대 담론의 가능성을 남태령 대첩이 보여주었다는 점에서, 남태령은 향후 시민사회 운동의 전략 수립과 담론형성에 시사점을 던져줄 수 있을 것으로 기대된다.
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
