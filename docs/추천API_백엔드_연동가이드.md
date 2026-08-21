# 추천 API 백엔드 연동 가이드

충전소 추천 모델을 백엔드(`ev-daegue_eta_api`)에 붙이는 방법입니다.

## 0. 왜 모델 파일이 아니라 API인가

모델 입력 25개 중 절반이 **DB 시계열에서 계산되는 값**입니다
(`avail_ratio_60m`, `changes_30m`, `time_since_charge_ended` …).
요청 payload만으로는 만들 수 없어서 `.joblib` 파일만 받아도 쓸 수 없습니다.

그렇다고 DB 접속정보와 서빙 코드 일체를 넘기면 모델을 재학습할 때마다 백엔드도
같이 배포해야 하고, 피처 스키마가 어긋나면 예외가 아니라 **조용히 틀린 예측**이
나옵니다. 그래서 경계를 HTTP로 긋습니다.

**백엔드는 이 API를 호출만 하면 됩니다. 모델이 바뀌어도 백엔드 코드는 그대로입니다.**

---

## 1. 접속 정보

```
POST http://<SERVER_IP>:8000/api/v1/chargers/recommend    추천 (인증 필요)
GET  http://<SERVER_IP>:8000/health                       상태 확인 (인증 불필요)
GET  http://<SERVER_IP>:8000/docs                         Swagger UI (브라우저 테스트용)
```

- **API 키는 이 문서에 없습니다.** 담당자에게 별도로 받으세요.
  요청 헤더 `X-API-Key: <키>` 로 보냅니다. 없거나 틀리면 **401**입니다.
- 현재 8000 포트는 인터넷 전체에 열려 있고 **키가 유일한 자물쇠**입니다.
  git·메신저 공개 채널에 올리지 마세요.
- 백엔드 서버 공인 IP가 정해지면 방화벽을 그 IP로 제한할 예정입니다.
  IP가 바뀌면 담당자에게 알려주세요.

---

## 2. 요청

> **필드 정본은 `/openapi.json` 입니다.** 서버가 요청 모델에서 직접 뽑으므로 항상
> 실제와 일치합니다. 아래 표는 시작용 요약이고, 빠진 필드가 있으면 스펙이 맞습니다.

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `dest_lat` | number | **Y** | — | 도착 위도 |
| `dest_lng` | number | **Y** | — | 도착 경도 |
| `eta_minutes` | number | **Y** | — | 예상 소요(분). 1 이상. 모델 horizon 입력 |
| `radius_km` | number | N | 2.0 | 검색 반경. 0.1 이상 |
| `top_k` | int | N | 10 | 추천 개수. 1~50 |
| `arrival_at` | string | N | null | 도착 예정 시각 ISO8601 |
| `mode` | `external`\|`home` | N | external | `home`은 등록 아파트 위주 |
| `registered_stat_ids` | string[] | `home`일 때 Y | null | 우리집/등록 아파트 |
| `origin_lat`, `origin_lng` | number | N | null | 우회 계산용 출발지 |
| `current_soc` | number | N | null | 현재 SOC %. 0~100. **현재 점수·제외에 쓰이지 않는다**(아래 참고) |
| `vehicle_model_id` | string | N | null | 차량 카탈로그 ID |
| `min_output_kw` | number | N | null | 최소 출력 하드 필터 |
| `include_slow` | bool | N | false | false면 급속만 추천 |

최소 요청은 `dest_lat`, `dest_lng`, `eta_minutes` 셋입니다.

```json
{ "dest_lat": 35.84217, "dest_lng": 128.68043, "eta_minutes": 15,
  "radius_km": 2, "top_k": 5, "current_soc": 60 }
```

---

## 3. 응답

> **필드 정본은 `/openapi.json` 입니다.** 2026-08-19 부터 응답도 서버가 스스로
> 기술합니다(`recommend_api/response_schema.py`). 전체 필드 목록·타입·설명은
> 거기서 보시고, 이 절은 **스펙만 봐서는 틀리게 쓰기 쉬운 것**만 다룹니다.

**응답 키가 snake_case입니다.** 백엔드는 CamelModel(camelCase) 규칙이므로
그대로 프론트에 흘리면 API 컨벤션이 깨집니다. `app/schemas/`에 매핑 모델을 두세요.

**키가 없는 것과 값이 `null` 인 것은 다릅니다.** `parking`·`stale_note`·
`pred_remaining_min` 같은 조건부 필드는 해당 없으면 **키 자체가 빠집니다.**
반대로 `addr`·`output_kw` 처럼 값만 `null` 인 필드는 키가 남습니다. 조건부
필드는 `null` 체크가 아니라 **키 존재 여부**로 분기하세요.

### 자주 쓰는 필드

| 필드 | 설명 |
|---|---|
| `recommendation_score` | **0~100 최종 추천점수** |
| `recommendation_label` | 매우 추천 / 추천 / 조건부 추천 / 주의·대안 부족 / 추천 어려움 |
| `avg_available_prob` | 충전소 평균 모델 확률. 추천점수와는 **별개 축**입니다 |
| `pred_available` / `total_chargers` | 도착시점 사용가능 예측 대수 / 호환 충전기 수 |
| `badges` | 충전소 단위 요약 플래그. 충전기 목록을 안 써도 경고를 띄울 수 있습니다 |
| `chargers[]` | 충전기별 도착시점 확률(`available_prob`)·상태·지연 여부 |
| `parking` | 주차장 정보. **매칭된 충전소에만 붙는 선택 필드**(아래 별도 절) |
| `meta.model_version` | 어느 모델이 답했는지. **로그에 남겨두면 나중에 성능 비교에 쓸 수 있습니다** |
| `meta.request_id` | 요청 추적용 UUID. 서버 예측 로그와 대조 가능 |
| `meta.confidence_level` | `high`(ETA ≤15분) / `medium`(16–20) / `low`(≥21) |

표시 밴드: ≥80 매우추천 · 65–79 추천 · 50–64 조건부 · 35–49 주의 · <35 추천어려움

### 조심할 필드 (값은 오는데 뜻이 다릅니다)

이 다섯 개는 이름만 보고 쓰면 **거짓 정보를 표시하게 됩니다.**

| 필드 | 함정 |
|---|---|
| `score_breakdown.battery` | **항상 `0.0`.** 배점에서 뺐고 계약 호환으로 키만 남겼습니다. 항목별 그래프에서 제외하세요 |
| `score_breakdown.parking_occupancy_coefficient` | **항상 `1.0` 이고 점수에 곱해지지 않습니다.** 주차 점유율과 충전기 가용률의 잔차 상관이 −0.073(R² 0.54%)이라 2026-08-04 에 순위 반영을 껐습니다. 만차는 **UI 안내로만** 막을 수 있습니다 |
| `detour_minutes` | `origin_lat`/`origin_lng` 를 **안 보내면 우회가 아닙니다.** 목적지↔충전소 직선거리를 30km/h로 환산한 값입니다 |
| `extra_distance_km` | 같은 조건에서 **`distance_m / 1000` 과 완전히 같은 값**입니다. "추가 거리"로 표시하면 거짓입니다 |
| `arrival_soc_pct` | **응답에 나오지 않습니다.** 도착 예상 SOC 를 점수·제외에서 뺐기 때문입니다. 소비자 스키마에 남겨두면 "언젠가 오는 값"으로 오해됩니다 — 지우세요 |

`detour_minutes`·`extra_distance_km` 를 제대로 쓰려면 요청에 `origin_lat`·
`origin_lng`(현위치)를 같이 보내세요. 그러면
`(출발→충전소 + 충전소→목적지) − (출발→목적지)` 로 실제 우회가 계산됩니다.

`remaining_model` 은 잔여시간 모델 **로드 여부**일 뿐 점수와 무관합니다
(2026-08-12 부터 `REMAINING_BLEND_WEIGHT=0`). `false` 면
`pred_remaining_min`·`free_by_eta_score` 표시값만 폴백으로 바뀝니다.
`free_by_eta_score` 는 ETA30 에서 거의 전부 `1.0` 으로 포화하므로
"곧 빈다"는 근거로 쓰지 마세요.

### `parking` (선택 필드, 2026-08-04 추가)

충전소가 공영·백화점 주차장 **부지 안에 있을 때만** 붙습니다. 없으면 키 자체가
없으니 `null` 체크가 아니라 **키 존재 여부**로 분기하세요.

```json
"parking": {
  "pklt_id": "155-3-000006",
  "parking_nm": "수성구청 주차장",
  "total_spaces": 142,
  "remaining_spaces": 27,
  "occupancy_rate": 80.43,
  "congestion_status": "혼잡(점유 90%미만)",
  "fee_type": "미상",
  "is_24h": false
}
```

`congestion_status` 는 **원본 그대로의 한글 문구**라 UI 에 바로 쓸 수 있습니다.
다만 출처 데이터라 값 집합이 보장되지 않고 `null` 일 수도 있으니, 직접 분기해야
한다면 `occupancy_rate` 를 쓰는 편이 안전합니다. `fee_type` 도 `"유료"`/`"무료"`/
`"유료+무료"`/`"미상"` 이 섞여 있습니다.

**커버리지가 낮고 지역 편중이 심합니다.** 전체 충전소 4,157곳 중 136곳(3.3%)만
매핑돼 있습니다. 실측 기준 top-10 추천에서 보통 0~3건에 붙습니다.

| 지역 | 반경 2km 급속 | `parking` 있음 |
|---|---|---|
| 성서산업단지 | 14곳 | 28.6% |
| 동성로 | 33곳 | 15.2% |
| 동대구역 | 37곳 | 10.8% |
| 칠곡(북구) | 13곳 | **0%** |

**필수 정보로 다루지 마세요.** "있으면 보여주고 없으면 숨기는" UI 여야 합니다.
없는 걸 "정보 없음"으로 표시하면 화면 대부분이 그 문구로 찹니다.

**가장 값이 큰 건 만차 경고입니다.** 점유율 90% 이상인 관측이 10.7% 라, 추천된
충전소 10곳 중 1곳꼴로 주차장이 거의 찬 상태입니다. **충전기가 비어 있어도 주차장이
만차면 진입을 못 합니다.** 이건 모델이 예측하지 못하는 부분이라(주차 점유율과 충전기
가용성의 잔차 상관 R² 0.54%) UI 안내로만 막을 수 있습니다.

**점수에는 반영되지 않습니다.** `score_breakdown.parking_occupancy_coefficient`
가 응답에 있지만 항상 `1.0` 이고 추천점수에 곱해지지 않습니다. 한때 구간별
계수(<50% → 1.0 / 50~70 → 0.9 / 70~90 → 0.75 / ≥90 → 0.5)를 곱했으나,
주차 점유율과 충전기 가용률의 시간대 제거 후 잔차 상관이 −0.073(R² 0.54%)로
예측 신호가 없어 2026-08-04 에 껐습니다. 충전 구획은 대개 전용이라 주차장이
만차여도 충전기는 비어 있습니다 — 그래서 **순위가 아니라 진입 안내**의 문제입니다.

`occupancy_rate` 는 원본이 비면 `1 − remaining/total` 로 계산해 채웁니다. 하필
만차인 주차장에서 원본이 비는 경향이 있어서 넣은 폴백입니다.

---

## 3-1. 랭킹 규칙

점수가 어떻게 나오는지는 `/openapi.json` 이 알려주지 못합니다. 여기가 정본입니다
(정확한 값은 `recommend_api/scoring.py` · `access.py` · `config.py`).

**3단계로 매깁니다.**

**① 후보 제외** — 6절의 세 가지 사유 중 하나라도 걸리면 목록에서 뺍니다.

**② 기본점수 100점** — 2026-08-18 기준 배분입니다.

| 항목 | 만점 | 기준 |
|---|---|---|
| 가용확률 | **50** | 충전소 평균 도착시점 확률 × 50 (선형) |
| 우회 시간 | 15 | 구간별 15 / 12 / 8 / 4 / 0 |
| 추가 거리 | 5 | 구간별 5 / 4 / 2 / 0 |
| 충전기 수 | **15** | 호환 대수별 15 / 12 / 9 / 5. 0대면 **제외** |
| 속도 적합 | 10 | `external`: 초급속10·급속9·중속6·완속3 / `home`: 완속10·중속6·급속3·초급속2 |
| 상태 최신성 | 5 | 갱신 ≤5분 5 / ≤10분 4 / ≤20분 2 / 그 외 0. 60분↑ **제외** |

> 7월 배분(가용확률45 · 배터리10 · 충전기수10)에서 바뀌었습니다. **배터리 항목은
> 제거**됐고 그 10점이 가용확률(+5)과 충전기 수(+5)로 갔습니다. 합계는 그대로
> 100 이라 표시 밴드는 영향받지 않습니다. `score_breakdown.battery` 는 호환용
> `0.0` 입니다.

**③ 접근성 계수** — 기본점수에 곱합니다.

| 구분 | 계수 |
|---|---|
| PUBLIC | ×1.00 |
| PUBLIC + 공동주택 부지 | ×0.90 |
| UNKNOWN | ×0.60 |
| 등록 아파트(`registered_stat_ids`) | ×1.00 |
| RESIDENT · RESTRICTED (비등록) | **제외** |

`home` 모드는 `registered_stat_ids` 안에서만 찾고 완속에 높은 배점을 줍니다.

**반경 자동 확장** — 최고점이 50 미만이면 반경을 1km 넓혀 재조회합니다
(최대 +2km, 절대 상한 5km). 넓혔으면 `meta.radius_expanded=true` 와
`meta.radius_note` 가 붙으니 "범위를 넓혀 찾았다"고 안내하세요.

---

## 4. 백엔드 연동 (2026-08-18 기준 **이미 붙어 있음**)

`ev-daegue_eta_api` 에 아래 3곳이 모두 반영돼 있습니다
(`app/domains/recommendations/` 의 `router.py` · `client.py` · `schema.py`).
아래는 신규 환경을 세우거나 연동이 깨졌을 때 대조할 기준입니다.

### ① `.env.example` + 각자의 `.env`

```bash
# --- 충전소 추천 모델 API (별도 서버) ---
RECOMMEND_API_BASE_URL=http://<SERVER_IP>:8000
RECOMMEND_API_TIMEOUT=10
RECOMMEND_API_KEY=
```

### ② `app/core/config.py` — `Settings` 클래스

```python
    # 추천 모델 API (외부 서버)
    recommend_api_base_url: str = "http://<SERVER_IP>:8000"
    recommend_api_timeout: float = 10.0
    recommend_api_key: str = ""
```

`tmap_app_key`, `data_go_kr_key`와 같은 자리에 두면 됩니다.

### ③ `app/domains/recommendations/router.py` + `client.py`

`APIRouter(prefix="/api/v1/recommendations", ...)` 에 **프리픽스가 이미 있으므로
경로는 `""`** 입니다 (안 그러면 `/api/v1/recommendations/recommend` 가 됩니다).
업스트림 호출은 `client.py` 로 분리돼 있습니다. 아래는 그 골자입니다.

```python
import httpx
from fastapi import HTTPException
from app.core.config import get_settings

@router.post("")
async def recommend(body: RecommendRequestIn):
    s = get_settings()
    payload = {
        "dest_lat": body.dest_lat,
        "dest_lng": body.dest_lng,
        "eta_minutes": body.eta_minutes,
        "radius_km": body.radius_km or 2,
        "top_k": body.top_k or 10,
        "current_soc": body.current_soc,
    }
    headers = {"X-API-Key": s.recommend_api_key} if s.recommend_api_key else {}
    try:
        async with httpx.AsyncClient(timeout=s.recommend_api_timeout) as c:
            r = await c.post(
                f"{s.recommend_api_base_url}/api/v1/chargers/recommend",
                json=payload, headers=headers,
            )
            r.raise_for_status()
    except httpx.TimeoutException:
        raise HTTPException(504, "추천 서버 응답 지연")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            raise HTTPException(500, "추천 서버 인증 실패 — API 키 확인")
        raise HTTPException(502, f"추천 서버 오류: {e.response.status_code}")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"추천 서버 통신 실패: {e}")
    return r.json()
```

위 예시는 payload 를 손으로 골라 담지만, 실제 구현은
`body.model_dump(by_alias=False, exclude_none=True)` 로 **요청 필드를 통째로
전달**합니다. 그쪽이 낫습니다 — 새 요청 필드가 생겨도 백엔드를 안 고쳐도 됩니다.

### 같이 확인할 것

- **라우터 등록** — `include_router`가 돼 있는지. 없으면 만들어도 404입니다
- **`by_alias=False`** — 업스트림은 **snake_case** 를 받습니다. camelCase 로 보내면
  필수 필드 누락으로 422 입니다
- **`httpx`** — `requirements.txt`에 있는지 (TMAP 호출에 이미 쓰면 그대로)

---

## 5. 타임아웃을 10초 이상으로

실측값입니다.

| 요청 | 첫 호출 | 이후 |
|---|---|---|
| 반경 2km · top 5 | 3.5초 | 0.4~1.4초 |
| 반경 5km · top 20 | 4.8초 | 2.7~2.8초 |
| `/health` | — | 0.06초 |

첫 호출이 느린 건 모델 로드·DB 커넥션 워밍업 때문입니다. 반경과 `top_k`가 커질수록
후보가 늘어 선형에 가깝게 느려집니다. **기본 3~5초 타임아웃이면 넓은 반경에서
끊깁니다.**

---

## 6. 에러 처리

| 상태 | 의미 | 대응 |
|---|---|---|
| 401 | API 키 없음/틀림 | 키 확인. 재시도해도 안 됨 |
| 400 | `mode=home`인데 `registered_stat_ids` 없음 | 요청 수정 |
| 503 | 서버에 모델 파일 없음 | 담당자 연락 |
| 504 (백엔드측) | 타임아웃 | 타임아웃 상향 또는 `radius_km`·`top_k` 축소 |

추천이 **0건**으로 오는 경우도 있습니다. 반경 내 후보가 전부 하드 제외된
상황입니다. 제외 사유는 현재 셋뿐입니다:

| 사유 | 뜻 |
|---|---|
| `compatible_chargers_zero` | 규격·출력 조건을 만족하는 충전기가 0대 |
| `status_very_stale` | 상태 갱신이 60분 이상 지연 |
| `access_restricted` | 입주민 전용 등 접근 불가(비등록) |

에러가 아니므로 UI에서 "조건에 맞는 충전소 없음"으로 처리하세요.

---

## 7. `/health` 모니터링

```json
{ "status": "prototype_validated",  // "degraded" 면 추천 후보가 줄어든 상태
  "model_version": "20260818T022438Z",
  "auth_required": true,
  "feature_coverage": 0.9287,       // 0.9 미만이면 degraded
  "feature_lag_min": 8,
  "code_fingerprint": "…",          // 배포 코드 내용 해시 (드리프트 확인용)
  "code_file_count": 40 }
```

`status`가 `degraded`면 **에러 없이 추천 후보만 조용히 줄어듭니다.**
서버 쪽 피처 생성 배치가 밀렸다는 뜻이라 담당자에게 알려주세요.

---

## 8. 테스트

Swagger UI에서 바로 호출해볼 수 있습니다 — `http://<SERVER_IP>:8000/docs`

curl:

```bash
curl -X POST http://<SERVER_IP>:8000/api/v1/chargers/recommend \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <키>" \
  -d '{"dest_lat":35.84217,"dest_lng":128.68043,"eta_minutes":15,"radius_km":2,"top_k":5}'
```

### 스펙은 서버에서 받으세요

```bash
curl -s http://<SERVER_IP>:8000/openapi.json -o recommend-openapi.json
```

요청·응답 스키마 **정본은 이 파일**입니다. 서버가 실제 코드에서 뽑아내므로
구현과 어긋날 수 없습니다. 클라이언트 타입이 필요하면 여기서 생성하세요
(`openapi-typescript`, `datamodel-code-generator` 등).

> **`api_contract/` 폴더는 2026-08-19 에 폐기했습니다.** 손으로 쓴 사본이라
> 2026-08-03 이후 갱신되지 않은 채 구현만 움직여 필드 10개가 빠지고
> `confidence_level` enum 에서 `medium` 이 누락돼 있었습니다. 그 폴더를 참고 중이면
> 즉시 위 스펙으로 갈아타세요.

`recommendations` 가 **빈 배열**로 오는 경우가 정상 응답에 포함됩니다(6절 참고).
스펙만 보고 UI 를 짜기 전에 그 경우를 먼저 처리하세요.

---

## 9. 모델이 갱신되면

담당자가 서버에서 모델 파일만 교체하고 API를 재시작합니다(10초 내외).

**백엔드는 아무것도 바뀌지 않습니다.** 계약이 그대로이기 때문입니다.
`meta.model_version`만 새 값으로 바뀌므로, 로그에 남겨두면 언제 모델이 바뀌었고
그 전후로 지표가 어떻게 달라졌는지 추적할 수 있습니다.
