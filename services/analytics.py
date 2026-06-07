import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..core.config import AnalyticsConfig, analytics_config
from ..repositories.repo import UserRepository, PostRepository, InteractionRepository

logger = logging.getLogger(__name__)


@dataclass
class EngagementStats:
    user_id: str
    username: str
    date: str
    views: int
    likes: int
    comments: int
    shares: int
    avg_dwell_ms: float
    engagement_rate: float
    rolling_7d_engagement: float

    def __repr__(self):
        return (
            f"[{self.date}] {self.username:20s} "
            f"ER={self.engagement_rate:.2f}% "
            f"7d_avg={self.rolling_7d_engagement:.2f}%"
        )


@dataclass
class CohortRow:
    cohort_month: str
    cohort_size: int
    months_since_signup: int
    active_users: int
    retention_pct: float

    def __repr__(self):
        bar = "█" * int(self.retention_pct / 5)
        return (
            f"Cohort {self.cohort_month} | "
            f"Ay+{self.months_since_signup:02d} | "
            f"{self.retention_pct:5.1f}% {bar}"
        )


@dataclass
class ChurnRecord:
    user_id: str
    username: str
    last_active_at: datetime
    sessions_last_30d: int
    sessions_prev_30d: int
    churn_risk_score: float
    churn_label: str       # "churned" | "at_risk" | "declining" | "healthy"

    def __repr__(self):
        emoji = {"churned": "💀", "at_risk": "⚠️", "declining": "📉", "healthy": "✅"}.get(
            self.churn_label, "?"
        )
        return (
            f"{emoji} {self.username:20s} "
            f"risk={self.churn_risk_score:.3f} "
            f"[{self.churn_label}]"
        )


@dataclass
class InfluenceRecord:
    user_id: str
    username: str
    direct_influence: float
    second_hop_influence: float
    influence_score: float
    influence_rank: int

    def __repr__(self):
        return (
            f"#{self.influence_rank:4d} {self.username:20s} "
            f"score={self.influence_score:.3f} "
            f"(direct={self.direct_influence:.2f}, hop2={self.second_hop_influence:.2f})"
        )


# ════════════════════════════════════════════════════════════
# ANALYTICS SERVICE
# ════════════════════════════════════════════════════════════
class AnalyticsService:
    """
    Engagement analizi, cohort analizi, churn tespiti
    ve network/graph analizlerini yürüten servis katmanı.
    """

    def __init__(self, session: Session, config: AnalyticsConfig = analytics_config):
        self.session = session
        self.cfg = config
        self.user_repo = UserRepository(session)
        self.post_repo = PostRepository(session)
        self.interaction_repo = InteractionRepository(session)

    # ════════════════════════════════════════════════════════
    # 1. ENGAGEMENT ANALİZİ
    # ════════════════════════════════════════════════════════
    def daily_engagement(
        self, since_days: int = 30, user_id: str = None
    ) -> list[EngagementStats]:
        """
        Günlük engagement istatistikleri + 7 günlük kayan ortalama.
        user_id verilirse sadece o kullanıcı için filtreler.
        """
        logger.info("[ANALYTICS] daily_engagement hesaplanıyor")

        params = {"since": datetime.utcnow() - timedelta(days=since_days)}
        user_filter = "AND p.user_id = :uid" if user_id else ""
        if user_id:
            params["uid"] = user_id

        rows = self.session.execute(text(f"""
            WITH daily AS (
                SELECT
                    p.user_id,
                    DATE_TRUNC('day', i.created_at)              AS day,
                    COUNT(*) FILTER (WHERE i.type='like')        AS likes,
                    COUNT(*) FILTER (WHERE i.type='comment')     AS comments,
                    COUNT(*) FILTER (WHERE i.type='share')       AS shares,
                    COUNT(*) FILTER (WHERE i.type='view')        AS views,
                    AVG(i.dwell_time_ms)                         AS avg_dwell
                FROM interactions i
                JOIN posts p ON p.id = i.post_id
                WHERE i.created_at >= :since {user_filter}
                GROUP BY p.user_id, day
            )
            SELECT
                d.user_id,
                u.username,
                TO_CHAR(d.day, 'YYYY-MM-DD')                    AS date,
                d.views, d.likes, d.comments, d.shares,
                COALESCE(d.avg_dwell, 0)                        AS avg_dwell_ms,
                -- Engagement rate
                CASE WHEN d.views = 0 THEN 0 ELSE
                    (d.likes + d.comments * 2.0 + d.shares * 3.0)
                    / d.views * 100
                END                                             AS engagement_rate,
                -- 7 günlük kayan ortalama
                AVG(
                    CASE WHEN d.views = 0 THEN 0 ELSE
                        (d.likes + d.comments * 2.0 + d.shares * 3.0)
                        / d.views * 100
                    END
                ) OVER (
                    PARTITION BY d.user_id
                    ORDER BY d.day
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                )                                               AS rolling_7d
            FROM daily d
            JOIN users u ON u.id = d.user_id
            ORDER BY d.user_id, d.day
        """), params).mappings().all()

        results = [
            EngagementStats(
                user_id=r["user_id"],
                username=r["username"],
                date=r["date"],
                views=r["views"],
                likes=r["likes"],
                comments=r["comments"],
                shares=r["shares"],
                avg_dwell_ms=float(r["avg_dwell_ms"] or 0),
                engagement_rate=float(r["engagement_rate"] or 0),
                rolling_7d_engagement=float(r["rolling_7d"] or 0),
            )
            for r in rows
        ]
        logger.info(f"[ANALYTICS] {len(results)} günlük satır döndü")
        return results

    def hourly_heatmap(self, since_days: int = 90) -> list[dict]:
        """Gün×Saat bazlı etkileşim yoğunluğu (heatmap verisi)."""
        rows = self.session.execute(text("""
            SELECT
                EXTRACT(DOW  FROM created_at)::INT  AS day_of_week,
                EXTRACT(HOUR FROM created_at)::INT  AS hour_of_day,
                COUNT(*)                             AS interaction_count,
                AVG(dwell_time_ms)                  AS avg_dwell_ms,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (
                    PARTITION BY EXTRACT(DOW FROM created_at)
                )                                   AS pct_of_day
            FROM interactions
            WHERE created_at >= NOW() - INTERVAL :days
            GROUP BY day_of_week, hour_of_day
            ORDER BY day_of_week, hour_of_day
        """), {"days": f"{since_days} days"}).mappings().all()

        days = ["Pazar","Pazartesi","Salı","Çarşamba","Perşembe","Cuma","Cumartesi"]
        return [
            {
                "day": days[r["day_of_week"]],
                "hour": r["hour_of_day"],
                "count": r["interaction_count"],
                "avg_dwell_ms": float(r["avg_dwell_ms"] or 0),
                "pct_of_day": float(r["pct_of_day"] or 0),
            }
            for r in rows
        ]

    # ════════════════════════════════════════════════════════
    # 2. COHORT ANALİZİ
    # ════════════════════════════════════════════════════════
    def cohort_retention(self) -> list[CohortRow]:
        """Aylık kullanıcı retention cohort analizi."""
        logger.info("[ANALYTICS] cohort_retention hesaplanıyor")

        rows = self.session.execute(text("""
            WITH cohorts AS (
                SELECT id AS user_id,
                       DATE_TRUNC('month', created_at) AS cohort_month
                FROM users
            ),
            activity AS (
                SELECT p.user_id,
                       DATE_TRUNC('month', i.created_at) AS activity_month
                FROM interactions i
                JOIN posts p ON p.id = i.post_id
                GROUP BY p.user_id, activity_month
            ),
            cohort_activity AS (
                SELECT
                    c.cohort_month,
                    a.activity_month,
                    COUNT(DISTINCT c.user_id) AS active_users,
                    (EXTRACT(YEAR  FROM AGE(a.activity_month, c.cohort_month)) * 12
                     + EXTRACT(MONTH FROM AGE(a.activity_month, c.cohort_month)))::INT
                        AS months_since_signup
                FROM cohorts c
                JOIN activity a ON a.user_id = c.user_id
                    AND a.activity_month >= c.cohort_month
                GROUP BY c.cohort_month, a.activity_month
            ),
            cohort_sizes AS (
                SELECT cohort_month, COUNT(*) AS cohort_size FROM cohorts GROUP BY cohort_month
            )
            SELECT
                TO_CHAR(ca.cohort_month,'YYYY-MM')  AS cohort_month,
                cs.cohort_size,
                ca.months_since_signup,
                ca.active_users,
                ROUND(ca.active_users * 100.0 / cs.cohort_size, 1) AS retention_pct
            FROM cohort_activity ca
            JOIN cohort_sizes cs ON cs.cohort_month = ca.cohort_month
            ORDER BY ca.cohort_month, ca.months_since_signup
        """)).mappings().all()

        return [
            CohortRow(
                cohort_month=r["cohort_month"],
                cohort_size=r["cohort_size"],
                months_since_signup=r["months_since_signup"],
                active_users=r["active_users"],
                retention_pct=float(r["retention_pct"]),
            )
            for r in rows
        ]

    # ════════════════════════════════════════════════════════
    # 3. CHURN ANALİZİ
    # ════════════════════════════════════════════════════════
    def churn_risk_scores(self, label_filter: str = None) -> list[ChurnRecord]:
        """
        Kullanıcı churn risk skorları.
        label_filter: "churned" | "at_risk" | "declining" | "healthy" | None (hepsi)
        """
        logger.info("[ANALYTICS] churn_risk_scores hesaplanıyor")

        rows = self.session.execute(text("""
            WITH ua AS (
                SELECT
                    u.id, u.username, u.last_active_at, u.avg_engagement_rate,
                    COUNT(s.id) FILTER (
                        WHERE s.started_at >= NOW() - INTERVAL '30 days'
                    )                                          AS sess_30,
                    COUNT(s.id) FILTER (
                        WHERE s.started_at BETWEEN NOW() - INTERVAL '60 days'
                                               AND NOW() - INTERVAL '30 days'
                    )                                          AS sess_prev
                FROM users u
                LEFT JOIN user_sessions s ON s.user_id = u.id
                GROUP BY u.id, u.username, u.last_active_at, u.avg_engagement_rate
            )
            SELECT
                id, username, last_active_at,
                sess_30, sess_prev,
                ROUND(
                    (1 - sess_30::FLOAT / NULLIF(sess_prev,0)) * 0.5
                    + LEAST(
                        EXTRACT(EPOCH FROM NOW() - last_active_at) / 2592000.0,
                        1
                      ) * 0.5
                , 3) AS churn_score,
                CASE
                    WHEN sess_30 = 0                      THEN 'churned'
                    WHEN sess_30 < sess_prev * 0.5        THEN 'at_risk'
                    WHEN sess_30 < sess_prev              THEN 'declining'
                    ELSE                                       'healthy'
                END AS churn_label
            FROM ua
            ORDER BY churn_score DESC NULLS LAST
        """)).mappings().all()

        results = [
            ChurnRecord(
                user_id=r["id"],
                username=r["username"],
                last_active_at=r["last_active_at"],
                sessions_last_30d=r["sess_30"] or 0,
                sessions_prev_30d=r["sess_prev"] or 0,
                churn_risk_score=float(r["churn_score"] or 0),
                churn_label=r["churn_label"],
            )
            for r in rows
            if label_filter is None or r["churn_label"] == label_filter
        ]
        logger.info(f"[ANALYTICS] {len(results)} kullanıcı churn analizi tamamlandı")
        return results

    # ════════════════════════════════════════════════════════
    # 4. NETWORK / GRAPH ANALİZİ
    # ════════════════════════════════════════════════════════
    def influence_scores(self, top_n: int = 50) -> list[InfluenceRecord]:
        """
        2-hop PageRank benzeri etki skoru.
        Doğrudan takipçi ağırlığı + 2. derece bağlantı ağırlığı.
        """
        logger.info("[ANALYTICS] influence_scores hesaplanıyor")

        rows = self.session.execute(text("""
            WITH direct AS (
                SELECT following_id AS user_id,
                       SUM(interaction_weight) AS direct_w
                FROM follows GROUP BY following_id
            ),
            hop2 AS (
                SELECT f2.following_id AS user_id,
                       SUM(f1.interaction_weight * f2.interaction_weight) AS hop2_w
                FROM follows f1
                JOIN follows f2 ON f2.follower_id = f1.following_id
                GROUP BY f2.following_id
            )
            SELECT
                u.id, u.username,
                COALESCE(d.direct_w, 0)  AS direct_influence,
                COALESCE(h.hop2_w, 0)    AS hop2_influence,
                COALESCE(d.direct_w, 0) * 0.7
                    + COALESCE(h.hop2_w, 0) * 0.3 AS influence_score,
                DENSE_RANK() OVER (
                    ORDER BY COALESCE(d.direct_w, 0) * 0.7
                           + COALESCE(h.hop2_w,  0) * 0.3 DESC
                ) AS influence_rank
            FROM users u
            LEFT JOIN direct d ON d.user_id = u.id
            LEFT JOIN hop2   h ON h.user_id = u.id
            ORDER BY influence_score DESC
            LIMIT :n
        """), {"n": top_n}).mappings().all()

        return [
            InfluenceRecord(
                user_id=r["id"],
                username=r["username"],
                direct_influence=float(r["direct_influence"]),
                second_hop_influence=float(r["hop2_influence"]),
                influence_score=float(r["influence_score"]),
                influence_rank=r["influence_rank"],
            )
            for r in rows
        ]

    def ml_feature_store(self, user_ids: list[str] = None) -> list[dict]:
        """
        Model eğitimi için hazır feature vektörü.
        pandas DataFrame'e doğrudan yüklenebilir.
        """
        logger.info("[ANALYTICS] ml_feature_store sorgusu çalışıyor")

        uid_filter = "AND u.id = ANY(:uids)" if user_ids else ""
        params = {}
        if user_ids:
            params["uids"] = user_ids

        rows = self.session.execute(text(f"""
            WITH fs AS (
                SELECT
                    u.id                                           AS user_id,
                    u.follower_count,
                    u.following_count,
                    u.avg_engagement_rate,
                    u.follower_count::FLOAT
                        / NULLIF(u.following_count, 0)            AS follow_ratio,
                    COUNT(s.id) FILTER (
                        WHERE s.started_at >= NOW() - INTERVAL '7 days'
                    )                                             AS sessions_7d,
                    AVG(s.session_duration_s) FILTER (
                        WHERE s.started_at >= NOW() - INTERVAL '7 days'
                    )                                             AS avg_session_7d,
                    COUNT(DISTINCT p.id)                          AS total_posts,
                    AVG(i.dwell_time_ms)                          AS avg_dwell_ms,
                    EXTRACT(EPOCH FROM NOW() - u.created_at)
                        / 86400                                   AS account_age_days
                FROM users u
                LEFT JOIN user_sessions s ON s.user_id = u.id
                LEFT JOIN posts p ON p.user_id = u.id
                LEFT JOIN interactions i ON i.user_id = u.id
                {uid_filter}
                GROUP BY u.id, u.follower_count, u.following_count,
                         u.avg_engagement_rate, u.created_at
            )
            SELECT *,
                LN(1 + follower_count)  AS log_followers,
                LN(1 + total_posts)     AS log_posts
            FROM fs
            ORDER BY user_id
        """), params).mappings().all()

        return [dict(r) for r in rows]
