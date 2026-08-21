# CLAUDE.md

EV 충전기 가용성 예측 스케줄러. 대구(zcode=27) 충전기 상태를 수집해 ETA 시점 가용
여부를 예측한다.

## 실행 위치 — 전부 서버 Docker다

상시 파이프라인은 **로컬이 아니라 Lightsail 서버(`<SERVER_IP>`, `~/scheduler`)의
docker compose**에서 돈다. 로컬에서 스크립트를 돌리는 건 일회성 분석·학습뿐이다.

| 서비스 | 컨테이너 | 주기 |
|---|---|---|
| `run.py` | `charger-status-scheduler` | 5분 (변경분) |
| `run_status_snapshot.py` | `charger-status-snapshot` | **6시간** (전량) |
| `build_features.py` | `charger-features-builder` | 4분 |
| `prune_status.py` | `charger-status-prune` | 24시간 (**45일** 롤링) |
| `prune_parking.py` | `parking-prune` | 24시간 (원본 7일 / 상태 45일) |
| `recommend_api.main:app` | `charger-recommend-api` | uvicorn, 상주 |
| `backfill_outcomes.py` | `charger-backfill` | 1시간 (예측로그 실측 라벨링) |

수집기 중 **`~/parking-collector/` 만 이 저장소 밖**이다(호스트 crontab, 10분 주기).
저장소 소유자 본인 것이며 팀원 것이 아니다 — 초기에 오인했다. 다만 정리 로직이
없어서 지우는 쪽은 `parking-prune` 이 맡는다.

SSH는 `ssh ev-aws`로 붙는다(`~/.ssh/config`). **Git Bash 의 ssh 는 호스트를 못 찾으니
PowerShell 로 쓸 것.** 명령에 `$(...)`·중괄호·중첩 따옴표를 넣으면 PowerShell 이
먼저 먹어치우므로, 복잡하면 나눠서 실행하는 편이 빠르다.

- **소스가 이미지에 구워져 있다.** 호스트 파일만 고치면 컨테이너는 그대로다.
  반영하려면 `docker compose build` → `docker compose up -d --no-deps <service>`.
- 반영 확인: `docker exec <container> grep -c "<식별자>" <경로>`
- 서비스명을 명시해서 올려라. 그냥 `up -d` 하면 compose의 **모든** 서비스가 뜬다.
- 호스트 python3에는 `pymysql`이 없다(pyenv). 정상이다 — `parking-collector` 외에는
  호스트에서 아무것도 안 돈다. `pip3 install` 하지 말 것.

## DB 제약 — 2GB 공용 서버다

`team_5` MariaDB가 위 컨테이너들과 **같은 2GB 박스**에 산다. `parking-collector` 도
같은 DB를 쓴다.

- **보존 정책이 없으면 무한히 는다.** parking 계열은 2026-08-04 까지 정책이 하나도
  없어 105MB/일로 늘고 있었다(EV 74MB/일보다 빠르다). 새 테이블을 만들면 `prune_*`
  에 등록했는지 확인할 것. 현재 EV 3.3GB + parking 1.8GB 에서 안정.

- 현재 설정(`/etc/mysql/mariadb.conf.d/99-tuning.cnf`): `innodb_buffer_pool_size=512M`,
  `key_buffer_size=16M`, swap 2GB, `vm.swappiness=10`
- **버퍼 풀은 한 번 정하고 끝나는 값이 아니다.** 컨테이너 구성이 바뀌면 다시 잡아라.
  2026-08-03 에 `flask-app`(360MB)을 비운 여유로 768M 까지 올렸는데, 다음 날
  `api`(143MB)+`features`(192MB)를 올리면서 그 여유가 사라져 512M 로 되돌렸다.
- 스왑 스래싱은 **InnoDB 지표로는 안 잡힌다.** 768M 일 때 히트율 99%대에
  `Innodb_buffer_pool_reads` 도 정상인데, 실제로는 `mariadbd` 가 RSS 436MB /
  Swap 757MB 로 풀 절반 이상이 디스크에 있었다. 판정은 `vmstat 1 5` 의 `si`
  (스왑 인)로 한다 — 0 이 아니면 풀이 큰 것이다(당시 1MB/s, io wait 3~4%).
- **대량 INSERT는 반드시 청크 커밋**(500행)에 청크 사이 `sleep`을 둘 것.
  25,385행을 단일 트랜잭션으로 넣었다가 2026-07-31 DB가 35분씩 두 번 멈추고
  재시작됐다. 같은 DB에 쓰는 5분 수집이 함께 굶는다.
- 전량 스냅샷을 6시간보다 자주 돌리지 말 것. 10분 주기 = 370만행/일 = 15.2GB/21일.
- 무거운 학습을 서버에서 돌리지 말 것. 여유 RAM이 300MB 남짓이라 OOM으로 DB가 같이 죽는다.
- `SET GLOBAL`은 안 된다(`<db_user>`에 SUPER 없음). my.cnf + 재시작만 가능.

## 데이터 의미 — delta는 '변경분'이다

`ev_charger_status.source`가 `'delta'`(5분 변경분) / `'snapshot'`(6시간 전량)을 가른다.

- **delta는 상태가 바뀐 충전기만 준다.** 즉 행이 없다 = 상태가 그대로였다.
  "관측이 없으니 모른다"로 다루면 안 된다 — LOCF(마지막 상태 유지)로 복원한다.
- 이벤트 행을 그냥 세서 가용률을 구하면 부풀려진다(실측 71.4% vs 실제 66% 수준).
  반드시 LOCF로 점유 패널을 만든 뒤 집계할 것.
- delta만으로는 커버리지가 부족하다. 12일간 한 번도 상태가 안 바뀐 충전기가 있어
  25,433대 중 76.8%만 관측됐다. snapshot이 그 앵커 역할을 한다.
- LOCF 정확도(2026-07-31 스냅샷 정답 대비): staleness 8시간 이내 97.7%,
  12~24시간 73.2%로 붕괴. 그래서 `LABEL_MAX_STALENESS_MIN=480`.

## 학습 — 진입점과 함정

```bash
py -m recommend_api.train                  # 정본 학습 (model_store.py엔 __main__ 없음)
py -m recommend_api.train --staged --work-dir F:/tmp/WORK   # 저메모리 경로 (현재 사실상 필수)
py -m recommend_api.train --max-rows 400000  # 빠른 확인용
py scripts/etl/run_status_snapshot.py      # 1회 수집 후 종료 (argparse 없음)

# staged work-dir(parts+memmap) 재사용. 재전개 없고 **DB 접속이 0 이다**
py scripts/analysis/rapid_thresholds_staged.py --work-dir DIR    # 급속 임계값
py scripts/analysis/walkforward_staged.py --work-dir DIR --write-metrics  # rolling
py scripts/analysis/station_breakdown_staged.py --work-dir DIR   # 충전소 분해
```

- **`load_joined(limit=N)`은 무작위 표본이 아니다.** `ORDER BY stat_id, chger_id`
  뒤에 자르므로 앞쪽 충전기만 담긴다(40만 행 = 9,707대 / 전체 = 약 2만 대).
  A/B 비교엔 써도 되지만 절대 수치로 읽지 말 것.
- 라벨은 `LABEL_METHOD="locf"`. 좌변(표본)은 delta만, **우변(라벨)은 delta+snapshot**
  (`load_label_series()`). snapshot을 표본에 넣으면 `hour`/`minute_slot`이
  "그 시각엔 전부 관측된다"를 외운다.
- 아티팩트를 덮어쓰기 전에 `recommend_api/artifacts/`를 백업할 것.
- 모델 교체 시 `config.py`의 `horizon_hgb.joblib (<version>)` 주석도 같이 갱신.
- LOCF 라벨은 관성(persistence) 성격이 있다. 평가 시 persistence 베이스라인과 비교할 것.
- **기본 경로는 이제 RAM 에 안 들어간다.** 2026-08-18 기준 base 2,106,246행 →
  지평샘플 14,498,475 · 설계행렬 float64 12.18GB. `--staged` 를 쓸 것.
- **`--matrix-dtype` 은 float64 가 기본이다(2026-08-18, float32 에서 변경).**
  sklearn 1.9.0 HGB 는 `check_array(dtype=[float64])` 라 float64 memmap 만
  제로카피로 읽는다. float32 면 적합 때 RAM 에 전량 업캐스트 복사한다 —
  실측 피크 23GB vs 11GB(커밋 40.89/41.85GB 까지 차서 가용 RAM 이 36MB 였다).
  디스크가 6.09→12.18GB 로 느는 대신이니 **디스크가 부족할 때만** float32.
  남은 피크는 `train_test_split`(`early_stopping='auto'` 가 비닝 전에 90% 복사)이고
  이건 dtype 과 무관하다. 상세는 `staged_training.py` 독스트링.
- **재학습으로 지표가 오르지 않는다 (2026-08-18 확인).** 8폴드 워크포워드에서
  학습행이 2.46M→13.79M(5.6배)인데 AUC 상관 −0.060 이고, 5일 학습(0.9783)이
  26일 학습(0.9756)보다 높았다. 구 20260810 모델을 같은 창에 태운 직접 대조도
  AUC +0.0002 · Brier −0.0002 로 무승부다(폴드 std 0.0038 의 0.05배).
  재학습은 **낡음 리셋(유지보수)** 이지 개선이 아니다. 승격 근거로 쓰지 말 것.
- 세그먼트 분석 계획이 있으면 **test 예측을 덤프**해 둘 것. 안 하면 자를 때마다
  홀드아웃 재적합이다(2026-08-18 에 25분·32분 두 번 낭비). 2,977,654행 예측이
  parquet 18.5MB 이고, 그걸로 잔차 분석이 6.8초에 끝났다.

## 평가 — 풀링 지표를 그대로 읽지 말 것 (2026-08-18 하루에 세 번 속았다)

지표를 움직인 것이 **모델이 아니라 모집단의 기저율**이었던 사례가 연달아 나왔다.

1. `date_holdout` 두 개를 비교했더니 AUC +0.0040 "개선". 실제로는 test 구간이
   08-07~10(기저 0.7581) vs 08-13~18(0.7736)이라 창이 쉬웠던 것. 같은 행에
   두 모델을 태우니 +0.0002 였다.
2. 급속 임계값 재산출에서 정책 C 가 h5 thr 을 0.27→0.96 으로 올리라고 했다.
   성능 변화가 아니라 급속 불가율이 10.48%→8.13% 로 떨어져 warn 예산 여유가
   1.14배→1.48배가 된 것이었다. 그래서 곡선을 **바꾸지 않았다**.
3. 충전소별 h30 불가Recall 최저 15곳이 전부 ME(환경부)라 "이 사업자가 문제"로
   보였다. 실제로는 ME 충전기가 93% 비어 있어서다(다른 사업자 ~50%). 불가율
   구간별로 갈라 보니 Recall 이 0.234(불가율<5%)→0.903(≥30%)으로 단조 상승,
   스피어만 rho +0.423(p=2.9e-23). **Recall 분산의 85.4%가 기저율로 설명된다.**

판정 규칙:

- 창 하나짜리 지표로 모델을 비교하지 말 것. 폴드 표준편차와 견줄 것
  (`walkforward_staged.py`). 8폴드 기준 AUC std 0.0038 이다.
- 세그먼트를 비교할 때는 **기저율로 잔차화**한 뒤 볼 것. 등장회귀로 기대 Recall 을
  빼면 명단이 통째로 바뀐다 — ME 편중(하위20%의 95.1%)이 사라지고 대신 평균
  82kW 급속 100곳이 남는다(전체 평균 31kW, 이들이 FN 의 39.3%).

## 서빙 — 백엔드에는 모델이 아니라 API를 넘긴다

모델 입력 25개 중 절반이 DB 시계열에서 나온다(`avail_ratio_60m`, `changes_30m`,
`time_since_charge_ended` …). 요청 payload 만으로는 못 만들므로 **joblib 만 넘기면
백엔드가 쓸 수 없다.** 그렇다고 DB 접속정보와 `service.py` 일체를 넘기면 재학습마다
백엔드를 같이 배포해야 하고, `feature_schema_hash` 가 어긋나면 예외가 아니라
**조용히 틀린 예측**이 나온다. 그래서 경계를 HTTP 로 긋는다.

```
POST /api/v1/chargers/recommend     스키마 정본: GET /openapi.json (손으로 쓴 api_contract/ 는 2026-08-19 폐기)
GET  /health                        status=degraded 면 피처 커버리지부터 볼 것
```

- 서빙은 최신 status 에 `ev_charger_features` 를 **INNER JOIN** 한다. `features` 가
  밀리면 에러 없이 **추천 후보만 조용히 줄어든다**. `/health` 의 `feature_coverage`
  가 0.9 미만이면 그 상태다.
- `artifacts/` 는 이미지에 굽지 않고 볼륨 마운트다. 재학습 후 joblib 만 서버에
  복사하고 `docker compose restart api` — 재빌드 불필요.
- **`REMAINING_MODEL_PATH` 는 이 저장소 밖**(`../EVCharger-model-test/`)을 기본값으로
  가리킨다. 컨테이너에선 항상 실패하는데 `remaining_time.py` 가 `None` 을 돌려주고
  넘어가므로 크래시가 없다. 모델을 `artifacts/` 에 두고 compose 에서 경로를 명시했다.
  2026-08-12 부터 `REMAINING_BLEND_WEIGHT=0` 이라 **점수에는 영향이 없고**,
  응답의 `pred_remaining_min` · `free_by_eta_score` 표시값만 폴백으로 바뀐다.
  확인은 `meta.remaining_model` 이 `true` 인지로.
- **`REMAINING_BLEND_WEIGHT` 는 0 이다 (2026-08-12 조치, 0.75 → 0).**
  잔여시간 성분이 HGB 를 망가뜨리고 있었다. 워크포워드 6폴드(수집 시작 07-22 ~
  08-12, 폴드마다 HGB 재학습)에서 **18셀 중 17셀의 최적 비중이 0**, 나머지도 0.1.
  폴드평균 Brier 악화폭은 ETA10 +9.8% · ETA20 +22.7% · **ETA30 +53.3%** 였다.
  캘리브레이션도 HGB 단독은 ETA30 에서 평균예측 0.472~0.478 vs 실제 0.470~0.484
  로 맞는데, 블렌드는 0.79 를 뱉었다. 배포 직전 운영 실측에서도 HGB 0.4499 인
  충전기가 `free_by_eta_score=1.0` 때문에 0.8625 로 나가고 있었다.
  **성분은 지우지 않았다** — `pred_remaining_min` · `free_by_eta_score` ·
  `meta.remaining_model` 은 응답 스키마에 있고 백엔드가 이미 읽으므로
  표시용으로 계속 나간다. 값이 틀린 게 아니라 점수에 실린 비중이 틀렸던 것이다.
  다시 올릴 거라면 성분을 조건부 종료확률(충전소×경과구간 해저드)로 바꾼 뒤
  w=0.1~0.3 · ETA≥20 한정이 최적이었다. 다만 이득이 Brier −0.0007 로 미미하다.
  근거표는 `config.py` 주석과 `data/reports/세션종료해저드_20260812.md` 8절,
  재현은 `py -m recommend_api.experiment_stat3_walkforward` (12분).
- **짧은 창의 효과 크기를 그대로 믿지 말 것.** 위 해저드 증분을 OOT 2.1일
  부트스트랩으로는 −0.0013/−0.0018 로 봤는데 워크포워드에서는 −0.0007/−0.0003
  이었다(2~5배 낙관). 부호는 맞았고 크기가 틀렸다. 승격 판단은 워크포워드로.
- **다만 `meta.remaining_model=true` 라고 안심하지 말 것.** 2026-08-12 실측에서
  운영 ML 경로도 ETA20 에서 persistence 와 동률(Brier 0.3448 vs 0.3426),
  ETA30 에서는 폴백과 구별되지 않았다(0.4917 vs 0.4904, 둘 다 persistence
  0.4568 보다 나쁨). 주범은 모델이 아니라 `free_by_eta_score` 다 — 점추정 잔여를
  ±10분 램프로 확률에 매핑하는데, ML 예측이 늘 10~25분이라 **ETA30 에서는 거의
  전부 1.0 으로 포화**한다. 실측 조건부 잔여는 경과 30분에서 중앙 15분 / 평균
  71분이라 점추정 자체가 불가능한 분포다. 부차적으로 아티팩트 `elapsed_grid` 가
  `[0..45]` 라 경과 45분 초과는 외삽이고 항상 ~15분을 돌려준다(경과 90분의 실측
  중앙 잔여는 135분). 상세는 `data/reports/세션종료해저드_20260812.md`.
- 의존성은 두 파일이다 — 루트는 ETL, `recommend_api/requirements.txt` 는 서빙(정본).
  Dockerfile 이 둘 다 설치한다. **`scikit-learn` 은 정확히 고정**한다(현재 1.9.0):
  joblib 피클은 학습 환경 버전에 묶여 있어 마이너 차이만으로 조용히 틀릴 수 있다.
- `.gitignore` 에 `*.joblib` 이 있어 모델은 git 으로 안 따라온다. 배포 시 별도 전송.
- **인증**: `RECOMMEND_API_KEY` 가 설정돼 있으면 `X-API-Key` 헤더를 검사한다(없으면
  무인증 — 기존 호출자 호환용). `/health` 는 모니터링용이라 항상 열려 있고
  `auth_required` 로 현재 상태를 노출한다. 8000 포트가 인터넷 전체에 열려 있어
  키가 사실상 유일한 자물쇠다.
- **DB 계정**: api 컨테이너만 `ev_model_reader`(SELECT + 예측로그 INSERT)를 쓴다.
  compose `api.environment` 에서 `DB_USER`/`DB_PASSWORD` 를 덮어쓴다. 수집 서비스는
  `.env` 의 공용 계정 그대로다(INSERT/DELETE 필요).
- **`ensure_table()` 함정**: `prediction_log.log_recommendations` 가 INSERT 전에
  `CREATE TABLE IF NOT EXISTS` 를 부르면, 테이블이 이미 있어도 MariaDB 가 CREATE
  권한을 검사해 1142 로 막는다. `try/except` 에 먹혀 **추천은 200 인데 로그만 조용히
  사라진다**. 서빙 경로에서는 기본으로 건너뛴다
  (`PREDICTION_LOG_ENSURE_TABLE=1` 로만 켤 것).
- **콜레이션**: `ev_charger_*` 는 `utf8mb4_general_ci`, `parking_*` 는
  `utf8mb4_unicode_ci` 다. 양쪽에 조인하는 테이블(`ev_charger_parking_map`)은
  컬럼별로 맞춰야 한다. 안 맞으면 1267 로 죽고, `parking.py` 가 그 예외를 삼켜
  **주차 정보만 조용히 사라진다**.
- 서빙 경로에서 지역 변수명은 `park_` 같은 접두사를 붙일 것. `total` 을 그대로 쓰면
  바깥의 호환 충전기 수를 덮어써 `score_station(n_compatible=None)` 으로 500 이 난다
  (2026-08-04 실제 장애).

## 코드 패턴

- 주석·docstring은 한국어. **"왜"를 실측 수치와 함께** 남기는 스타일이다
  (근거표·사고 타임라인까지 docstring에 박아둔다). 새 코드도 이 밀도를 맞출 것.
- 충전기별 파이썬 루프 대신 `merge_asof(by=["stat_id","chger_id"])`를 쓸 것.
  루프 구조는 14만 번 호출이라 96만 행에서 1시간이 넘고, `by=`는 17배 빠르다
  (결과는 행 단위로 동일함을 검증했다).
- `pymysql`은 기본 `autocommit=False`다. 폴링·모니터링 연결은 REPEATABLE READ
  스냅샷에 갇혀 **값이 얼어붙는다**. `autocommit=True`를 넘길 것.
- `pd.read_sql(conn)`의 SQLAlchemy 경고는 이 저장소 전반의 정상 동작이다.

## 환경

- Windows / PowerShell 주. 여러 줄 파이썬은 Bash 도구 + heredoc이 안전하다
  (PowerShell `-c` 안에서 SQL 따옴표가 깨진다).
- 임시 스크립트는 `data/reports/`나 스크래치패드에. `scripts/analysis/`는 일회성 분석용.

## 기각된 시도 (같은 길을 다시 걷지 말 것)

- **주차장 데이터 결합** — 2026-08-04 기각. 시간대 평균을 제거한 잔차 상관 −0.073
  (R² 0.54%). 표본 부족이 아니다: 매칭된 130곳은 거리 중앙값 40m 에 이름까지 일치하는
  *같은 부지 안 충전기* 라, 가설이 맞다면 거기서 가장 강해야 했다. 충전 구획은 대개
  전용이라 주차장이 만차여도 충전기는 비어 있다. 커버리지가 늘어도 뒤집히기 어렵다.
  다만 **표시용**으로는 살려 뒀다(만차 경고). 상세는
  `data/reports/parking_결합평가_20260804.md`.
- `is_operating_at_arrival`, 세션 요일 prior — `config.py` 의 `FEATURE_COLS` 주석 참고.
- **stat=3 세션 해저드의 '충전기별' 계층** — 2026-08-12 기각. 조건부 종료확률
  자체는 잔여시간 성분 단독으로는 크게 낫지만(현재 폴백 대비 Brier 0.4899→0.2270),
  **HGB 를 포함한 전체 점수에서는 증분이 ΔBrier −0.002 / ΔAUC +0.009 로 줄어든다**
  (OOT 2.1일, 충전소 클러스터 부트스트랩). HGB 가 `time_since_charge_started` ·
  `capped_state_duration` · `avail_ratio_*` 로 이미 그 신호를 갖고 있다.
  성분 단독 비교만 보고 판단하지 말 것. 그리고 사다리를
  *충전기* 부터 시작할 이유가 없다. 충전소 계층 위에서 증분이 0 이다
  (ETA10 −0.0018 · ETA20 +0.0007 · ETA30 +0.0006, 위약 대조 포함). 충전기별
  세션길이는 분할반분 0.54 로 실재하지만 그 안정성을 충전소가 이미 담고 있다.
  같은 충전소 충전기는 같은 이용자 집단을 받는다. 사다리는 **충전소 → 출력버킷
  → 전역×경과** 로 시작할 것. 상세는 `data/reports/세션종료해저드_20260812.md`,
  재현은 `py -m recommend_api.experiment_session_hazard`.

## 품질관리 루프 — 배선은 됐고 트래픽이 없다 (2026-08-12 확인)

- **오프라인 평가**: `ev_charger_status` 는 학습·평가에 제대로 쓰인다. 수집 시작
  (2026-07-22)부터의 워크포워드 평가도 `holdout_eval.evaluate_rolling_folds`
  (확장창, 테스트일마다 재학습, `min_train_days=2`)로 가능하다. **다만 그 함수는
  DataFrame 입력이라 현재 데이터량에서 OOM 이다.** 대신
  `scripts/analysis/walkforward_staged.py`(memmap 입력, 정책·출력 동일)를 쓸 것.
  2026-08-18 모델에는 8폴드(확장창·3일 간격·`min_train_days=5`, OOS 07-27~08-18,
  126분)를 돌려 `rolling` 섹션을 채웠다. 요약 AUC 0.9735±0.0038 ·
  Brier 0.0412±0.0042 · 불가Recall 0.8677±0.0077.
- **ETA별 폴드 평균이 병목을 가리킨다**: 불가Recall 이 h5 0.9684 → h30 0.8383 →
  h60 0.7458 이고 표준편차가 0.009 로 작다. 즉 특정 시기 운이 아니라 전 창에서
  일관된 구조적 한계다. 다만 h30 의 0.8383 은 풀링값이라 과소평가다 — 불가율
  30% 이상 충전소 437곳만 보면 0.903 이다(위 "평가" 절).
- **온라인 QC**: `ev_recommend_prediction_log` (테이블명 주의 —
  `prediction_log` 아니다). `charger-backfill` 이 1시간마다 실측 라벨을 채운다.
  배선은 정상이고 1,717행 중 1,653행(96.3%)이 라벨링돼 있다.
- **그런데 표본이 사실상 없다.** 2026-07-27~08-12 17일간 1,717건(≈100건/일)이고
  07-28~30 · 08-01~03 · 08-08~10 은 통째로 비어 있다. 실사용 트래픽이 아니라
  테스트 호출로 보인다. **이걸로는 지표를 못 낸다.**
- **`ops_metrics.py` 는 어디에도 스케줄돼 있지 않다.** compose 에 서비스가 없어
  쌓인 로그로 지표를 계산하는 주체가 없다. 돌리려면 수동이다.
- **선택편향 주의**: 로그에 남는 건 이미 추천 후보로 뽑힌 행이라 가용률이 높게
  나온다(ETA15 에서 0.90). 모집단 정확도로 읽으면 안 된다.
- `horizon_hgb_metrics.json` 의 `status` 가 `staged_retrain_not_promoted`
  ("승격 전 검토 필요")인데 그 아티팩트가 운영에서 돌고 있다
  (`/health` `model_version=20260810T095551Z` 일치). 승격 시 status 갱신이
  누락된 것으로 보이나, 확인 전에는 단정하지 말 것.

## 알려진 이슈

- DB 비밀번호가 약하다. 방화벽 IP 화이트리스트에 의존 중이라 언젠가 정리 필요.
- **로컬과 서버가 손으로 동기화된다.** 서버 `~/scheduler` 는 git 클론이 아니라
  파일만 올라간 상태다. 2026-08-03~04 에만 세 번 어긋났다(compose 의 `--interval 10`
  잔존, `recommend_api/` 디렉터리 자체 누락, 편집이 로컬에만 반영).
  `docker compose build` 가 전부 `0.0s` 캐시로 끝나면 반영이 안 된 것이다.

  **규칙 1 — 서버에는 `main` 에 머지된 것만 올린다.** 워크트리가 여럿이라
  "로컬"이 어느 브랜치인지가 매번 다르다. 실제로 `F:/dev/scheduler` 체크아웃은
  `main` 이 아니라 `claude/model-bench-app-dashboard` 였다(2026-08-12 확인).
  대조 기준이 흔들리면 대조 자체가 무의미하다.

  **규칙 2 — `recommend_api/` 는 디렉터리째 올린다.** 바뀐 파일만 골라 올리면
  서버 트리가 어느 커밋과도 일치하지 않게 되고, 아래 지문 대조가 깨진다.

  **드리프트 확인은 코드 지문으로 한다** (2026-08-12 추가):

  ```bash
  py scripts/code_fingerprint.py                                  # 로컬
  curl -s localhost:8000/health | tr ',' '\n' | grep code_        # 서버
  ```

  두 `code_fingerprint` 가 같아야 한다. `recommend_api/**/*.py` 내용 해시라
  (개행 정규화 후) 커밋 해시 없이도 대조된다. `code_file_count` 가 다르면
  부분 복사다. 계산 규칙은 `recommend_api/code_stamp.py` docstring 참고.

  **`/health` 의 `git_commit` 은 드리프트 확인에 쓸 수 없다.** 그건 배포 코드가
  아니라 **모델 학습 시점** 커밋이고(아티팩트에 박혀 있다), 2026-08-12 확인 시점
  값이 `8090f93` = "first commit" 으로 재빌드해도 안 바뀌었다. `code_commit`
  (빌드 인자로 박는 주장값)과 `code_fingerprint`(내용 해시)를 볼 것.
- `api`(143MB)·`features`(192MB) 가 상주하면서 2GB 가 빠듯하다. 컨테이너를 더
  붙이려면 버퍼 풀을 다시 줄이거나 인스턴스를 4GB 로 올려야 한다.
- `pandas`/`numpy` 는 버전 범위가 열려 있다. 현재 컨테이너는 pandas 3.0.5 /
  numpy 2.4.6 / scipy 1.17.1 이며 모델 언피클에 문제없음을 확인했다.
- **`weather` 는 수동 운영이다 (2026-08-18 결정).** `scripts/etl/run_weather.py`
  가 수집기인데 `docker-compose.yml` 에 서비스가 없다. **compose 에 추가하지 말 것**
  — 2GB 박스에 컨테이너를 더 붙이는 대신 사용자가 명령으로 직접 돌리기로 했다.

  ```bash
  py scripts/etl/run_weather.py --once       # 전일 24시간 UPSERT (평소)
  py scripts/etl/run_weather.py --backfill   # 최근 14일 일괄 (며칠 밀렸을 때)
  ```

  - **인자 없이 실행하면 `schedule` 데몬으로 떠서 안 끝난다.** 반드시 둘 중 하나를 줄 것.
  - `.env` 도 cwd 도 안 탄다. DB 접속정보와 API 키가 스크립트에 하드코딩돼 있어
    (`.env` 의 `KMA_SERVICE_KEY` 는 이 스크립트가 안 읽는다) 어디서 실행해도 같다.
    키 자체는 유효함을 2026-08-18 에 확인했다.
  - `--backfill` 은 최근 14일 고정이다. 더 긴 공백은 `collect_day()` 를 날짜
    범위로 직접 부를 것(2026-08-18 에 07-31~08-17 18일을 그렇게 메웠다 — 432행,
    학습구간 648행 누락 0). UPSERT(PK `observed_at,stn_id`)라 재실행이 안전하다.
  - 당일 자료는 못 받는다 — API 가 `전날 자료까지 제공됩니다`(코드 99)로 거절한다.
    그래서 최신 데이터는 항상 전일까지다.
  - **아직 `FEATURE_COLS` 에 없다.** 쓰려면 `rn` 은 결측이 "비 안 옴"이므로
    `fillna(0)`(`WEATHER_DEFAULTS` 전제와 동일), `dsnw` 는 여름 구간 전량 NULL 이라
    정보량이 0 이다. 관측소도 대구 1곳(143)뿐이라 충전소별 차이는 안 잡힌다.
