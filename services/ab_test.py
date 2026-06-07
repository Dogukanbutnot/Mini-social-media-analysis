import math
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..repositories.repo import RecommendationRepository

logger = logging.getLogger(__name__)


@dataclass
class ModelStats:
    model_version: str
    total_recommendations: int
    total_clicks: int
    ctr_pct: float
    avg_score: float
    avg_virality_on_click: float
    avg_dwell_after_click: float
    model_rank: int

    def __repr__(self):
        bar = "█" * int(self.ctr_pct / 2)
        return (
            f"#{self.model_rank} [{self.model_version:10s}] "
            f"CTR={self.ctr_pct:.2f}% {bar} "
            f"| recs={self.total_recommendations} clicks={self.total_clicks}"
        )


@dataclass
class ABTestResult:
    version_a: str
    version_b: str
    ctr_a: float
    ctr_b: float
    ctr_lift_pct: float
    z_score: float
    p_value: float
    is_significant: bool       # p < 0.05
    winner: str

    def __repr__(self):
        sig = "✅ ANLAMLI" if self.is_significant else "⚠️ YETERSİZ ÖRNEKLEM"
        return (
            f"A/B Test: {self.version_a} vs {self.version_b}\n"
            f"  CTR: {self.ctr_a:.2f}% → {self.ctr_b:.2f}% "
            f"(lift={self.ctr_lift_pct:+.2f}%)\n"
            f"  z={self.z_score:.3f} p={self.p_value:.4f} → {sig}\n"
            f"  Kazanan: {self.winner}"
        )


@dataclass
class PrecisionAtK:
    k: int
    precision: float
    total_hits: int
    total_recs: int
    users_evaluated: int

    def __repr__(self):
        return (
            f"Precision@{self.k} = {self.precision:.4f} "
            f"({self.total_hits}/{self.total_recs} tıklama, "
            f"{self.users_evaluated} kullanıcı)"
        )


# ════════════════════════════════════════════════════════════
# AB TEST & MODEL MONITORING SERVICE
# ════════════════════════════════════════════════════════════
class ABTestService:
    """
    Model sürümlerini karşılaştır, istatistiksel anlamlılık hesapla,
    öneri kalitesini izle (CTR, Precision@K, dwell after click).
    """

    def __init__(self, session: Session):
        self.session = session
        self.rec_repo = RecommendationRepository(session)

    # ════════════════════════════════════════════════════════
    # 1. MODEL PERFORMANS TABLOSU
    # ════════════════════════════════════════════════════════
    def model_performance(self, since_days: int = 14) -> list[ModelStats]:
        """
        Son N günde her model versiyonunun CTR, avg_score,
        tıklama sonrası virality ve dwell time metriklerini döner.
        """
        rows = self.session.execute(text("""
            SELECT
                r.model_version,
                COUNT(*)                                              AS total_recs,
                COUNT(*) FILTER (WHERE r.was_clicked)                 AS total_clicks,
                ROUND(
                    COUNT(*) FILTER (WHERE r.was_clicked) * 100.0
                    / NULLIF(COUNT(*), 0), 2
                )                                                     AS ctr_pct,
                AVG(r.score)                                          AS avg_score,
                AVG(p.virality_score) FILTER (WHERE r.was_clicked)    AS avg_virality,
                AVG(i.dwell_time_ms)  FILTER (WHERE r.was_clicked)    AS avg_dwell,
                RANK() OVER (ORDER BY
                    COUNT(*) FILTER (WHERE r.was_clicked) * 100.0
                    / NULLIF(COUNT(*), 0) DESC
                )                                                     AS model_rank
            FROM recommendations r
            JOIN posts p ON p.id = r.target_post_id
            LEFT JOIN interactions i
                ON i.post_id = r.target_post_id
                AND i.user_id = r.source_user_id
                AND i.type = 'view'
            WHERE r.created_at >= NOW() - INTERVAL :days
            GROUP BY r.model_version
            ORDER BY ctr_pct DESC
        """), {"days": f"{since_days} days"}).mappings().all()

        results = [
            ModelStats(
                model_version=r["model_version"],
                total_recommendations=r["total_recs"],
                total_clicks=r["total_clicks"],
                ctr_pct=float(r["ctr_pct"] or 0),
                avg_score=float(r["avg_score"] or 0),
                avg_virality_on_click=float(r["avg_virality"] or 0),
                avg_dwell_after_click=float(r["avg_dwell"] or 0),
                model_rank=r["model_rank"],
            )
            for r in rows
        ]
        logger.info(f"[AB] {len(results)} model versiyonu analiz edildi")
        return results

    # ════════════════════════════════════════════════════════
    # 2. A/B TEST İSTATİSTİKSEL KARŞILAŞTIRMA
    # ════════════════════════════════════════════════════════
    @staticmethod
    def _z_score_to_p(z: float) -> float:
        """İki kuyruklu normal dağılım p değeri (yaklaşık)."""
        # Abramowitz & Stegun yaklaşımı
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        coeffs = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
        poly = sum(c * t**i for i, c in enumerate(coeffs, 1))
        p_one_tail = poly * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        return min(2 * p_one_tail, 1.0)

    def ab_compare(
        self, version_a: str, version_b: str, since_days: int = 14
    ) -> ABTestResult:
        """
        İki model sürümü arasında istatistiksel anlamlılık testi.
        Proporsiyon z-testi kullanır.
        """
        row = self.session.execute(text("""
            SELECT
                SUM(CASE WHEN model_version = :va THEN 1 ELSE 0 END) AS n_a,
                SUM(CASE WHEN model_version = :va AND was_clicked THEN 1 ELSE 0 END) AS c_a,
                SUM(CASE WHEN model_version = :vb THEN 1 ELSE 0 END) AS n_b,
                SUM(CASE WHEN model_version = :vb AND was_clicked THEN 1 ELSE 0 END) AS c_b
            FROM recommendations
            WHERE created_at >= NOW() - INTERVAL :days
              AND model_version IN (:va, :vb)
        """), {"va": version_a, "vb": version_b, "days": f"{since_days} days"}).first()

        if not row:
            raise ValueError("Yeterli veri yok")

        n_a, c_a, n_b, c_b = int(row[0] or 0), int(row[1] or 0), int(row[2] or 0), int(row[3] or 0)

        if n_a == 0 or n_b == 0:
            raise ValueError(f"Model verisi eksik: n_a={n_a}, n_b={n_b}")

        p_a = c_a / n_a
        p_b = c_b / n_b
        p_pool = (c_a + c_b) / (n_a + n_b)

        std_err = math.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        z = (p_b - p_a) / std_err if std_err > 0 else 0.0
        p_value = self._z_score_to_p(z)
        is_sig = p_value < 0.05

        ctr_lift = (p_b - p_a) * 100

        winner = (
            version_b if (is_sig and p_b > p_a) else
            version_a if (is_sig and p_a > p_b) else
            "Fark anlamlı değil"
        )

        result = ABTestResult(
            version_a=version_a,
            version_b=version_b,
            ctr_a=round(p_a * 100, 3),
            ctr_b=round(p_b * 100, 3),
            ctr_lift_pct=round(ctr_lift, 3),
            z_score=round(z, 4),
            p_value=round(p_value, 4),
            is_significant=is_sig,
            winner=winner,
        )
        logger.info(f"[AB] {result}")
        return result

    # ════════════════════════════════════════════════════════
    # 3. PRECISION@K
    # ════════════════════════════════════════════════════════
    def precision_at_k(self, k: int = 10, since_days: int = 7) -> PrecisionAtK:
        """Son N günde verilen önerilerin Precision@K metriği."""
        row = self.session.execute(text(f"""
            WITH ranked AS (
                SELECT
                    source_user_id,
                    was_clicked,
                    ROW_NUMBER() OVER (
                        PARTITION BY source_user_id ORDER BY score DESC
                    ) AS rk
                FROM recommendations
                WHERE created_at >= NOW() - INTERVAL :days
            ),
            per_user AS (
                SELECT
                    source_user_id,
                    COUNT(*) FILTER (WHERE was_clicked AND rk <= :k) AS hits,
                    COUNT(*) FILTER (WHERE rk <= :k)                 AS shown
                FROM ranked
                GROUP BY source_user_id
            )
            SELECT
                AVG(hits::FLOAT / NULLIF(shown, 0)) AS precision,
                SUM(hits)   AS total_hits,
                SUM(shown)  AS total_shown,
                COUNT(*)    AS users_evaluated
            FROM per_user
        """), {"days": f"{since_days} days", "k": k}).first()

        result = PrecisionAtK(
            k=k,
            precision=round(float(row[0] or 0), 4),
            total_hits=int(row[1] or 0),
            total_recs=int(row[2] or 0),
            users_evaluated=int(row[3] or 0),
        )
        logger.info(f"[AB] {result}")
        return result

    # ════════════════════════════════════════════════════════
    # 4. MODEL MONITORING DASHBOARD (özet rapor)
    # ════════════════════════════════════════════════════════
    def monitoring_report(self, since_days: int = 14) -> dict:
        """
        Tüm model metriklerini tek sözlükte döner.
        Günlük cron job veya Slack bildirimi için kullanılabilir.
        """
        perf = self.model_performance(since_days)
        p_at_k = self.precision_at_k(k=10, since_days=7)

        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "period_days": since_days,
            "models": [
                {
                    "version": m.model_version,
                    "ctr_pct": m.ctr_pct,
                    "total_recs": m.total_recommendations,
                    "avg_score": round(m.avg_score, 4),
                    "avg_dwell_ms": round(m.avg_dwell_after_click, 0),
                    "rank": m.model_rank,
                }
                for m in perf
            ],
            "precision_at_10": p_at_k.precision,
            "best_model": perf[0].model_version if perf else None,
            "best_ctr": perf[0].ctr_pct if perf else 0.0,
        }

        # En iyi vs en kötü modeli karşılaştır
        if len(perf) >= 2:
            try:
                ab = self.ab_compare(
                    perf[-1].model_version, perf[0].model_version, since_days
                )
                report["ab_test"] = {
                    "version_a": ab.version_a,
                    "version_b": ab.version_b,
                    "lift_pct": ab.ctr_lift_pct,
                    "is_significant": ab.is_significant,
                    "winner": ab.winner,
                    "p_value": ab.p_value,
                }
            except ValueError as e:
                report["ab_test"] = {"error": str(e)}

        return report
