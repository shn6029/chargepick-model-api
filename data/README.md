# data/

원본 파일과 임시 리포트 저장소. 대용량·개인키는 git에 올리지 않습니다.

## 하위 폴더

| 경로 | 내용 |
|------|------|
| [raw/](raw/) | 원본 csv/xlsx (gitignore) |
| [processed/](processed/) | 생성 학습셋 parquet (gitignore) + meta json |
| [reports/](reports/) | 분석 로그·임시 리포트 |

### processed/

`scripts/etl/build_station_timeseries.py` 산출물. parquet 은 46MB라 gitignore 하고
`station_training_v2_meta.json`(행수·피처·소스 해시)만 추적한다. 없으면 ETL 재실행.

| 파일 | 내용 |
|------|------|
| `station_timeseries_features.parquet` | 충전소 × tick 시계열 피처 8종 (698만 행) |
| `station_horizon_training_v2.parquet` | DA① v1 학습셋 + 시계열 피처 (161만 행 × 42컬럼) |
| `station_training_v2_meta.json` | 생성 시각·소스 해시·결측률 |

정본 평가 문서·모델은 `recommend_api/artifacts/`에 둡니다. 여기는 중간 산출물용입니다.
