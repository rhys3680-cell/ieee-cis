# IEEE-CIS Fraud Detection

Kaggle의 **IEEE-CIS Fraud Detection** 데이터를 분석하기 위한 프로젝트입니다. 원본 CSV를 DuckDB 웨어하우스로 통합하고, 이후 피처 분석과 사기 탐지 모델 학습에 사용할 수 있는 기반 데이터를 만듭니다.

## 구성

```text
data/
├── ieee-fraud-detection/    # Kaggle 원본 CSV
└── warehouse/
    └── warehouse.duckdb     # 생성된 DuckDB 웨어하우스
src/ieee_cis/
└── etl/                     # CSV 통합·정규화·파생 피처 생성
notebooks/                   # 탐색 및 피처 분석 노트북
tests/                       # ETL 단위 테스트
```

## 사전 준비

- Python 3.13 이상
- [uv](https://docs.astral.sh/uv/)
- Kaggle의 [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) 데이터

의존성을 설치합니다.

```bash
uv sync --all-groups
```

다운로드한 데이터에서 아래 파일을 `data/ieee-fraud-detection/`에 배치합니다.

```text
train_transaction.csv
train_identity.csv
test_transaction.csv
test_identity.csv
sample_submission.csv
```

## 웨어하우스 생성

다음 명령은 원본 CSV를 읽어 `data/warehouse/warehouse.duckdb`의 `txn` 테이블을 생성하거나 갱신합니다.

```bash
uv run python -m ieee_cis.etl.build_warehouse
```

처리 결과는 train과 test를 합친 1,097,231행의 `txn` 테이블입니다. 원본 데이터가 필요하고 실행에는 수 분이 걸릴 수 있습니다.

## 데이터 처리 내용

- `train_transaction.csv`와 `test_transaction.csv`를 하나의 테이블로 통합
- 각 split의 identity 데이터를 `TransactionID`로 LEFT JOIN
- test identity의 `id-01` ~ `id-38` 컬럼명을 train과 같은 `id_01` ~ `id_38`로 정규화
- test 행의 `isFraud`는 0이 아닌 `NULL`로 보존
- 다음 파생 컬럼 추가
  - `dataset_split`: `train` 또는 `test`
  - `has_identity`: identity 정보 연결 여부
  - `hour_slot`: `TransactionDT` 기준 24시간 상대 슬롯(0~23)
  - `day_index`: `TransactionDT` 기준 상대 일자

`TransactionDT`는 실제 날짜가 아닌 기준 시점으로부터의 초 단위 오프셋이므로, 시간 피처도 실제 시각이 아닌 상대 지표입니다.

## DuckDB 조회 예시

```python
import duckdb

con = duckdb.connect("data/warehouse/warehouse.duckdb", read_only=True)

con.execute("""
    SELECT dataset_split, COUNT(*) AS rows, AVG(isFraud) AS fraud_rate
    FROM txn
    GROUP BY dataset_split
""").df()
```

## 테스트

```bash
uv run pytest
```

테스트는 컬럼 타입 정의, identity 컬럼명 정규화, 시간 파생 피처, 생성 SQL의 문법을 검증합니다. 원본 CSV가 있으면 거래 CSV 헤더도 타입 정의와 대조합니다.

## 현재 범위

현재 구현 범위는 재현 가능한 데이터 적재와 기본 피처 생성입니다. 모델 학습, 시간 기준 검증, 예측 및 Kaggle 제출 파일 생성은 이후 단계에서 추가할 예정입니다.
