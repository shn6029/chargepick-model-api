"""충전소-주차장 매핑 / 실시간 혼잡도 조인 (추천점수용)."""

from __future__ import annotations

from typing import Any

from .model_store import get_connection


def load_latest_occupancy_for_stations(stat_ids: list[str]) -> dict[str, dict[str, Any]]:
    """매핑된 충전소의 최신 주차 혼잡도.

    Returns: {stat_id: {pklt_id, remaining_spaces, occupancy_rate, ...}}
    """
    if not stat_ids:
        return {}
    placeholders = ",".join(["%s"] * len(stat_ids))
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    m.stat_id, m.pklt_id, m.parking_nm, m.parking_total_spaces,
                    m.parking_fee_type, m.parking_24h,
                    r.remaining_spaces, r.occupancy_rate, r.congestion_status,
                    r.collected_at
                FROM ev_charger_parking_map m
                LEFT JOIN (
                    SELECT r1.*
                    FROM parking_realtime_status r1
                    INNER JOIN (
                        SELECT pklt_id, MAX(collected_at) AS max_at
                        FROM parking_realtime_status
                        GROUP BY pklt_id
                    ) latest
                      ON r1.pklt_id = latest.pklt_id
                     AND r1.collected_at = latest.max_at
                ) r ON m.pklt_id = r.pklt_id
                WHERE m.stat_id IN ({placeholders})
                """,
                tuple(str(s) for s in stat_ids),
            )
            cols = [d[0] for d in cur.description]
            out: dict[str, dict[str, Any]] = {}
            for row in cur.fetchall():
                d = dict(zip(cols, row))
                out[str(d["stat_id"])] = d
            return out
    except Exception:
        return {}
    finally:
        conn.close()
