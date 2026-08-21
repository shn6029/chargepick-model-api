-- 도착 시점 상태 조회 / 백필 / JOIN 로드 가속용
-- 현황: idx_stat_time(stat_id, chger_id, stat_upd_dt) 만 존재 → created_at 조합 인덱스 추가
-- 적용 전: SHOW INDEX FROM ev_charger_status;

CREATE INDEX idx_status_charger_time
ON ev_charger_status (
    stat_id,
    chger_id,
    created_at
);
