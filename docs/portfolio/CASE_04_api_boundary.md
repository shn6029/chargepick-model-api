# 케이스 스터디 4 — 모델이 아니라 API를 넘긴 이유

> **한 줄**: 팀에 무엇을 넘길지 정하는 것이 아키텍처 결정이었다. 경계를 HTTP로 긋고,
> 그 경계에서 발생하는 **조용한 실패**들을 하나씩 요란한 실패로 바꿨다.

| | |
|---|---|
| 경계 | `POST /api/v1/chargers/recommend` · `GET /health` |
| 스키마 정본 | 서버가 내는 `GET /openapi.json` (손으로 쓴 `api_contract/`는 2026-08-19 폐기) |
| 드리프트 검출 | `/health`의 `code_fingerprint` — 로컬과 curl 한 번으로 대조 |
| 회귀 테스트 | `test_response_schema.py` 21개 · 구현 소스를 **AST로 파싱** |
| 근거 | 커밋 `5cfe002` · [response_schema.py](../../recommend_api/response_schema.py) 독스트링 |

---

## 1. 무엇을 넘길 것인가

이 프로젝트는 팀 프로젝트다. 백엔드 팀원이 이 예측 결과를 소비해야 한다.
선택지가 셋 있었다.

### ① joblib 파일만 넘긴다 — 불가능하다

모델 입력 25개 중 **절반이 DB 시계열에서 나온다.**

```
avail_ratio_15m / 30m / 60m      최근 가용 비율
changes_30m                       최근 30분 상태 전환 횟수
time_since_available              마지막 가용 시점부터 경과
time_since_charge_started / ended 세션 경과
capped_state_duration             현재 상태 유지 시간
```

요청 payload에는 목적지 좌표와 ETA밖에 없다. 이 피처들은 **과거 상태 이력을
조회해야만** 만들 수 있다. joblib만 넘기면 백엔드가 쓸 방법이 없다.

### ② 코드와 DB 접속정보 일체를 넘긴다 — 위험하다

두 가지가 걸린다.

1. **재학습할 때마다 백엔드를 같이 배포해야 한다.** 모델 갱신 주기와
   백엔드 릴리스 주기가 묶인다.
2. 더 나쁜 건 — 모델의 `feature_schema_hash`와 피처 생성 코드가 어긋나면
   **예외가 아니라 조용히 틀린 예측**이 나온다. 컬럼 순서가 밀린 행렬도
   `predict_proba`는 숫자를 반환한다.

### ③ HTTP로 경계를 긋는다 — 채택

```
POST /api/v1/chargers/recommend
GET  /health
```

모델·피처·DB는 전부 이쪽 안에 갇힌다. 재학습은 `artifacts/` 볼륨에 joblib을
복사하고 `docker compose restart api`로 끝난다. **백엔드는 배포에 참여하지 않는다.**

---

## 2. 경계를 그었더니 새 문제 — 계약은 누가 지키나

API 경계를 정하면 **스키마 계약**이 필요하다. `api_contract/` 폴더에 OpenAPI YAML과
mock JSON을 손으로 써서 넣었다.

그리고 **2026-08-03 이후 한 번도 갱신되지 않았다.** 그 사이 `service.py`와
`scoring.py`는 08-18까지 계속 움직였다. 16일치 표류를 감사한 결과:

| 항목 | 실제 |
|---|---|
| 누락된 필드 | **10개** — 요청 `include_slow`, 응답 `parking`(8키), `meta` 4개, 충전기 2개, `score_breakdown` 2개 |
| `confidence_level` | 계약은 `enum: [high, low]` — **서버는 `medium`도 낸다**(ETA 16~20분). 엄격한 검증기라면 정상 응답을 거부한다 |
| 배점표 | 7월판(가용확률 45 · 배터리 10 · 충전기수 10) — 실제는 **50 / 0 / 15** |
| 제거된 항목 | `current_soc` "5% 미만 하드 제외", `arrival_soc_pct` — 구현에서 사라졌는데 계약에 남아 있음 |

가장 해로웠던 건 **mock JSON**이다. 구 공식으로 **자체 정합한 오답**이었다
(battery 10.0, charger_count 6.0 = 구 10점 눈금). 그리고 팀원 가이드 문서가
**프론트엔드에게 "이 mock으로 UI 작업하라"** 고 지시하고 있었다.
그대로 갔으면 **영원히 안 채워지는 게이지**를 만들었을 것이다.

### 근본 원인 — 계약 파일은 아무것도 실행하지 않는다

감사에서 결정적인 사실이 나왔다.

> **팀 저장소의 `api_contract` 참조가 0건이었다.**
> 백엔드의 `RecommendParking` 8개 키는 계약이 아니라 **실물 응답을 보고 맞춘 것**이었다.

계약이 정본이 아니라 **장식**이었다. 아무도 안 읽는 문서는 틀려도 아무도 모른다.

---

## 3. 해법 — 서버가 스스로 말하게 한다

사본을 다시 쓰는 대신, **모델이 곧 응답**이 되게 했다.

요청 스키마는 사실 이미 정확했다 — FastAPI가 `RecommendRequest`에서 자동으로
뽑아내고 있었고 `include_slow`도 들어 있었다. **손으로 베낀 사본만 틀렸다.**

못 뽑던 건 **응답뿐**이었다. 엔드포인트가 `response_model` 없이 raw dict을
반환해서 `/openapi.json`의 200 응답이 `"schema": {}`였다.

`response_schema.py`에 Pydantic 모델 7개를 만들어 붙였다.

```python
@app.post("/api/v1/chargers/recommend", response_model=RecommendResponse, ...)
```

> **이제 표류가 구조적으로 불가능하다. 응답을 바꾸려면 이 파일을 지나야 하기 때문이다.**

---

## 4. 그런데 `response_model`이 새 위험을 만든다

`response_model`을 붙이면 FastAPI가 응답을 모델로 필터링한다.
**모델에 없는 키는 예외 없이 잘려나간다.**

즉 `service.py`에 필드를 추가하고 `response_schema.py`를 깜빡하면,
**에러 없이 응답에서만 사라진다.** 이 저장소가 반복해서 당한 "조용히 틀림" 패턴 그대로다.

그래서 `extra="allow"`로 열어뒀다. 모델에 없는 키도 통과하므로 **소실이 원천 봉쇄된다.**

대신 스펙이 뒤처질 수 있다. 그 뒤처짐은 테스트가 잡는다.

> **조용한 데이터 손실을 요란한 테스트 실패로 바꾼 것이다.**

### 테스트가 소스를 AST로 읽는 이유

응답 전체를 만들려면 DB와 모델 아티팩트가 필요하다. 손으로 쓴 고정 픽스처를 두면
**그 픽스처가 `service.py`와 같이 낡는다** — 지금 고치고 있는 문제를 한 겹 아래에
다시 만드는 셈이다.

그래서 **구현 소스의 dict 리터럴 키를 직접 파싱**한다.

```python
tree = ast.parse((_HERE / "service.py").read_text(encoding="utf-8"))
```

누군가 `service.py`의 station dict에 `"foo": ...`를 추가하면, 이 테스트가 그 자리에서
"모델에 foo가 없다"고 실패한다. **응답을 한 번도 만들지 않고도 잡힌다.**

파서가 죽어서 공허하게 통과하는 것을 막는 가드도 넣었다 — 실제로 표본 사각지대를 한 번 잡았다.

---

## 5. `exclude_unset` — 와이어 포맷 자체가 계약이다

현재 응답은 "키가 아예 없는 것"과 "키는 있고 값이 `null`인 것"을 **의도적으로** 섞어 쓴다.

| | 필드 |
|---|---|
| 키 자체가 없음 | `parking`(주차 매칭 있을 때만) · `stale_note` · `pred_remaining_min` 3종(충전중일 때만) |
| 키는 있고 `null` | `addr` · `chger_type` · `output_kw` · `arrival_at` · `radius_note` |

연동가이드가 `parking`에 대해 **"`null` 체크가 아니라 키 존재 여부로 분기하라"** 고
명시했으므로 이 구분은 계약의 일부다.

`exclude_none=True`를 쓰면 `addr: null`까지 전부 사라져 **와이어가 바뀐다.**
`exclude_unset=True`는 입력 dict에 있던 키만 남기므로 현재 포맷이 키 단위로 보존된다.

```python
response_model_exclude_unset=True   # exclude_none 이 아니다
```

### 타입을 `Literal`로 좁히지 않은 이유

`confidence_level` · `recommendation_label` · `mode`는 값 집합이 정해져 있지만
`str`로 뒀다. `Literal`로 좁히면 값이 하나 늘어나는 순간 응답 검증이 실패해
**추천 엔드포인트가 500으로 죽는다.**

서빙 경로에서 문서 정확도를 위해 가용성을 거는 건 손해다.
값 목록은 각 필드 `description`에 적었다 — 2절의 `confidence_level` enum 오류가
정확히 이 반대 방향의 사고였다.

---

## 6. 조용한 실패 4종 — 이 시스템의 반복 패턴

이 프로젝트에서 가장 위험했던 버그들은 **예외를 던지지 않았다.**
전부 200 OK가 나가면서 내용만 틀렸다.

### ① 권한 1142 — 추천은 200인데 로그만 사라진다

`prediction_log.log_recommendations`가 INSERT 전에 `CREATE TABLE IF NOT EXISTS`를 불렀다.
**테이블이 이미 있어도 MariaDB는 CREATE 권한 자체를 검사한다.**
최소 권한 계정(SELECT + 해당 테이블 INSERT)으로 서빙하면 여기서 1142로 막히고,
바깥 `try/except`에 먹혀 조용히 사라진다.

→ **스키마 생성은 배치의 일**로 옮겼다. 서빙 경로에서는 기본으로 건너뛴다.

```python
if os.getenv("PREDICTION_LOG_ENSURE_TABLE", "0") == "1":
    ensure_table(conn)
```

### ② 콜레이션 1267 — 주차 정보만 사라진다

`ev_charger_*`는 `utf8mb4_general_ci`, `parking_*`는 `utf8mb4_unicode_ci`다.
양쪽에 조인하는 `ev_charger_parking_map`은 **컬럼별로 콜레이션을 맞춰야 한다.**
하나로 통일하면 반대쪽 조인이 죽는다.

```sql
stat_id  VARCHAR(20) NOT NULL COLLATE utf8mb4_general_ci,   -- ev_charger_* 쪽
pklt_id  VARCHAR(40) NOT NULL COLLATE utf8mb4_unicode_ci,   -- parking_* 쪽
```

`parking.py`가 이 예외를 삼키도록 돼 있어서, 안 맞으면 주차 정보만 조용히 빠진다.

### ③ INNER JOIN — 추천 후보만 조용히 줄어든다

서빙은 최신 status에 `ev_charger_features`를 **INNER JOIN**한다.
피처 빌더가 밀리면 에러가 아니라 **후보 수만 준다.**

`lag`(최신 시각)만으로는 부분 누락을 못 잡는다 — 배치가 최신 행 몇 개만 처리하고
나머지를 빠뜨려도 lag는 0으로 보인다. 그래서 **커버리지**를 따로 잰다.

```json
"feature_coverage": 0.9982,
"status": "degraded"        // coverage < 0.9 이면
```

### ④ 변수 섀도잉 — 실제 장애 (2026-08-04)

주차 혼잡도를 붙이면서 `total`이라는 지역 변수를 썼는데,
그 스코프의 `total`은 이미 **호환 충전기 수**였다. 덮어쓰는 순간
`score_station(n_compatible=None)`으로 500이 났다.

→ 서빙 경로에서는 `park_` 접두사를 규칙으로 정하고 주석에 박아뒀다.

```python
park_rem   = park.get("remaining_spaces")
park_total = park.get("parking_total_spaces")
```

---

## 7. 배포 드리프트 — `git_commit`은 쓸모가 없었다

서버 `~/scheduler`는 **git 클론이 아니라 파일만 손으로 올라간 상태**다.
"지금 서버가 어느 코드를 돌고 있나"를 물을 방법이 없었고,
2026-08-03~04 사흘 동안만 **세 번** 어긋났다(compose 잔존 인자, 디렉터리 통째 누락,
편집이 로컬에만 반영).

`/health`에 `git_commit`이 이미 있었지만 그건 **모델 학습 시점 커밋**이다
(joblib 아티팩트의 필드를 그대로 읽는다). 확인 시점의 값이 `8090f93` = "first commit"이었고
**재빌드해도 안 바뀌었다.** 드리프트 탐지에는 아무 쓸모가 없었다.

그래서 두 값을 추가했다.

| 필드 | 성격 |
|---|---|
| `code_commit` | 빌드 시 `--build-arg`로 박은 커밋. **주장값** — 안 주면 `unknown`, 잘못 주면 잘못 나온다 |
| `code_fingerprint` | 컨테이너 안 `recommend_api/*.py`를 실제로 읽어 만든 해시. **거짓말을 못 한다** |

```bash
py scripts/code_fingerprint.py
```

```bash
curl -s localhost:8000/health | grep code_fingerprint
```

두 값이 다르면 어긋난 것이다. `code_file_count`가 다르면 **부분 복사**다.

### 지문 계산에서 결정한 것 둘

- **`experiment_*.py`도 포함한다.** 서빙에 안 쓰지만, **부분 복사가 드리프트의
  원인이었으므로** "디렉터리 전체가 같은가"를 묻는 게 맞다. 그래서 배포 규칙도
  "`recommend_api/`는 디렉터리째 올린다"로 정했다.
- **개행을 정규화한다.** Windows 체크아웃은 CRLF, 컨테이너는 LF라
  정규화하지 않으면 **내용이 같아도 값이 갈린다.**

실제 배포에서 지문이 `03abeb0a4d19`/38개 → `453644518c70`/40개로 바뀌는 걸 확인했다.

---

## 8. 인증과 최소 권한

8000 포트가 인터넷 전체에 열려 있어 **API 키가 사실상 유일한 자물쇠**다.

```python
if not RECOMMEND_API_KEY:
    return                          # 키 미설정 시 통과 — 기존 호출자 호환
if not secrets.compare_digest(x_api_key, RECOMMEND_API_KEY):
    raise HTTPException(401, ...)
```

- `compare_digest`를 쓴다. `!=`는 앞에서부터 비교하다 다른 문자가 나오면 즉시 반환해서,
  **응답 시간 차이로 키를 한 글자씩 맞출 수 있다.**
- 키를 안 넣고 배포하면 무인증 상태다. 그 사실을 밖에서 알 수 있게
  `/health`가 `auth_required`로 **현재 상태를 노출**한다.
- `/health`는 모니터링용이라 항상 열어 뒀다.

DB도 나눴다. **api 컨테이너만** `ev_model_reader`(SELECT + 예측로그 INSERT)를 쓴다.
수집 서비스는 INSERT/DELETE가 필요해 공용 계정 그대로다.
6절 ①의 1142 사고가 바로 이 분리 때문에 드러난 것이다.

---

## 9. 정리

1. **무엇을 넘길지가 아키텍처 결정이다.** 모델 파일을 넘길 수 없는 이유(피처의 절반이
   DB 시계열)와 코드 일체를 넘기면 안 되는 이유(배포 결합 + 조용한 스키마 불일치)를
   먼저 정리하니 HTTP 경계가 답으로 나왔다.
2. **아무도 실행하지 않는 문서는 반드시 낡는다.** 손으로 쓴 계약은 16일 만에 필드 10개가
   어긋났고, 정작 팀은 그걸 안 읽고 실물 응답을 보고 맞추고 있었다.
   **서버가 스스로 스키마를 내게** 해서 표류를 구조적으로 막았다.
3. **안전장치가 새 위험을 만들 수 있다.** `response_model`은 스키마를 보장하는 대신
   조용한 필드 소실을 도입한다. `extra="allow"` + AST 회귀 테스트로
   **조용한 손실을 요란한 실패로** 바꿨다.
4. **와이어 포맷은 취향이 아니라 계약이다.** "키 없음"과 "값 null"의 구분이
   클라이언트 분기 조건이므로 `exclude_none`이 아니라 `exclude_unset`이어야 한다.
5. **가용성과 문서 정확도가 충돌하면 가용성이다.** `Literal`로 좁히면 값이 하나 느는 순간
   추천 엔드포인트가 500으로 죽는다.
6. **손으로 동기화하는 배포는 검증 수단을 함께 만든다.** 주장값(`code_commit`)과
   내용 해시(`code_fingerprint`)를 나눠 노출해 curl 한 번으로 대조된다.

---

## 재현

```bash
py -m pytest recommend_api/test_response_schema.py -q
```

```bash
py scripts/code_fingerprint.py
```

| 파일 | 내용 |
|---|---|
| [`response_schema.py`](../../recommend_api/response_schema.py) | 모델 7개 · `extra`/`exclude_unset`/`Literal` 결정 근거 |
| [`test_response_schema.py`](../../recommend_api/test_response_schema.py) | AST 기반 회귀 테스트 21개 |
| [`code_stamp.py`](../../recommend_api/code_stamp.py) | 지문 계산 규칙 |
| [`main.py`](../../recommend_api/main.py) | `/health` · 인증 · `response_model` 설정 |
| `docs/추천API_백엔드_연동가이드.md` | 백엔드용 연동 문서 |
| 커밋 `5cfe002` | `api_contract/` 폐기 · 16일 표류 감사 전문 |
