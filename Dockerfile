FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Seoul

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# 의존성은 두 파일로 나뉜다 — 루트는 수집·피처 ETL, recommend_api/ 는 서빙(FastAPI).
# 이미지는 모든 서비스(collector/snapshot/features/prune/api)가 공유하므로 둘 다 깐다.
# 한쪽에 몰아 적으면 sklearn 핀이 두 군데로 갈라져 어긋난다.
COPY requirements.txt ./
COPY recommend_api/requirements.txt ./requirements-api.txt
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

# 수집·피처 스크립트 일체. 서비스별 command 는 docker-compose.yml 에서 지정한다.
# (예전에는 저장소 루트의 run.py 하나만 복사했다 — 파일이 scripts/etl/ 로 이동됨)
COPY scripts/ ./scripts/

# 추천 API 서빙 코드. artifacts/ 도 같이 들어오지만(2MB 남짓) compose 에서
# 볼륨으로 덮어쓴다 — 재학습 때마다 이미지를 다시 굽지 않기 위해서다.
# 마운트가 없는 환경에서도 최소한 뜨도록 이미지에 굽는 쪽을 남겨 둔다.
COPY recommend_api/ ./recommend_api/

# 배포된 코드가 무엇인지 /health 로 확인하기 위한 스탬프.
# COPY 뒤에 두어야 커밋만 바뀔 때 앞 레이어 캐시가 안 깨진다.
# 서버 ~/scheduler 는 git 클론이 아니라 여기서 rev-parse 를 못 한다 —
# 빌드하는 쪽에서 `--build-arg CODE_COMMIT=$(git rev-parse HEAD)` 로 넘길 것.
# 안 넘기면 unknown 이고, 그때는 /health 의 code_fingerprint(내용 해시)로 본다.
ARG CODE_COMMIT=unknown
ENV CODE_COMMIT=$CODE_COMMIT

CMD ["python", "scripts/etl/run.py"]
