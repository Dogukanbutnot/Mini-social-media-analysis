import math
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..core.config import RecommendationConfig, rec_config
from ..repositories.repo import (
    UserRepository, PostRepository,
    InteractionRepository, FollowRepository, RecommendationRepository
)

logger = logging.getLogger(__name__)


# ── Veri sınıfları ──────────────────────────────────────────
@dataclass
class RecommendedPost:
    post_id: str
    author: str
    content_preview: str
    media_type: str
    score: float
    source: str          # "cf", "cbf", "hybrid", "cold_start", "trending"
    reason: str          # Kullanıcıya gösterilebilecek açıklama

    def __repr__(self):
        preview = self.content_preview[:60].replace("\n", " ")
        return (
            f"[{self.source.upper():10s}] score={self.score:.3f} "
            f"| {self.author}: {preview}..."
        )


@dataclass
class UserSimilarity:
    user_id: str
    username: str
    similarity: float
    common_items: int


# ════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE
# ════════════════════════════════════════════════════════════
class RecommendationEngine:
    """
    Hybrid recommendation engine.

    Desteklenen stratejiler:
        - user_cf   : User-based Collaborative Filtering (Pearson)
        - item_cf   : Item-based Collaborative Filtering (Cosine)
        - content   : Content-based Filtering (embedding + hashtag)
        - hybrid    : CF + CBF + Popularity birleşimi
        - cold_start: Yeni kullanıcı için trending içerikler
    """

    def __init__(self, session: Session, config: RecommendationConfig = rec_config):
        self.session = session
        self.cfg = config
        self.user_repo = UserRepository(session)
        self.post_repo = PostRepository(session)
        self.interaction_repo = InteractionRepository(session)
        self.follow_repo = FollowRepository(session)
        self.rec_repo = RecommendationRepository(session)

    # ── Yardımcı: sinyal ağırlığı ───────────────────────────
    def _signal_weight(self, interaction_type: str, dwell_ms: int = 0) -> float:
        base = self.cfg.signal_weights.get(interaction_type, 0.0)
        if interaction_type == "view":
            base += min(dwell_ms / 30_000, 1.0)
        return base

    # ── Yardımcı: min-max normalizasyon ─────────────────────
    @staticmethod
    def _minmax(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        lo, hi = min(scores.values()), max(scores.values())
        if hi == lo:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    # ── Yardımcı: tazelik faktörü (exponential decay) ────────
    def _freshness(self, created_at: datetime) -> float:
        hours_old = (datetime.utcnow() - created_at).total_seconds() / 3600
        half_life = self.cfg.cbf_freshness_half_life_hours
        return math.exp(-0.693 * hours_old / half_life)

    # ════════════════════════════════════════════════════════
    # 1. USER-BASED COLLABORATIVE FILTERING
    # ════════════════════════════════════════════════════════
    def user_cf(self, user_id: str, limit: int = 20) -> list[RecommendedPost]:
        """Pearson korelasyonu ile kullanıcı benzerliğine dayalı öneri."""
        logger.info(f"[CF] user_cf başlatıldı — user={user_id[:8]}")

        # Hedef kullanıcının etkileşim vektörü
        target_interactions = self.interaction_repo.get_user_interactions(
            user_id, since_days=90
        )
        if not target_interactions:
            logger.warning("[CF] Yeterli etkileşim yok → cold_start'a yönlendiriliyor")
            return self.cold_start(limit=limit)

        # {post_id: score}
        target_vector: dict[str, float] = {}
        for i in target_interactions:
            target_vector[i.post_id] = target_vector.get(i.post_id, 0) + \
                self._signal_weight(i.type, i.dwell_time_ms)

        target_mean = sum(target_vector.values()) / len(target_vector)

        # Komşu adayları: hedefin takip ettiklerinin takipçileri (2-hop)
        following_ids = self.follow_repo.get_following_ids(user_id)
        if not following_ids:
            return self.cold_start(limit=limit)

        # Komşuların vektörlerini SQL ile çek
        following_tuple = tuple(following_ids)
        rows = self.session.execute(text("""
            SELECT
                i.user_id,
                i.post_id,
                SUM(CASE i.type
                    WHEN 'share'  THEN 5.0
                    WHEN 'save'   THEN 4.0
                    WHEN 'like'   THEN 3.0
                    WHEN 'report' THEN -5.0
                    ELSE 1.0 + LEAST(i.dwell_time_ms / 30000.0, 1.0)
                END) AS score
            FROM interactions i
            WHERE i.user_id IN :uids
            GROUP BY i.user_id, i.post_id
        """), {"uids": following_tuple}).mappings().all()

        # Komşu vektörlerini oluştur
        neighbor_vectors: dict[str, dict[str, float]] = {}
        for row in rows:
            uid, pid, sc = row["user_id"], row["post_id"], float(row["score"])
            neighbor_vectors.setdefault(uid, {})[pid] = sc

        # Pearson korelasyonu hesapla
        similarities: list[UserSimilarity] = []
        for neighbor_id, neighbor_vec in neighbor_vectors.items():
            common = set(target_vector) & set(neighbor_vec)
            if len(common) < self.cfg.cf_min_common_items:
                continue

            n_mean = sum(neighbor_vec.values()) / len(neighbor_vec)

            num = sum(
                (target_vector[p] - target_mean) * (neighbor_vec[p] - n_mean)
                for p in common
            )
            den_t = math.sqrt(sum((target_vector[p] - target_mean) ** 2 for p in common))
            den_n = math.sqrt(sum((neighbor_vec[p] - n_mean) ** 2 for p in common))

            if den_t * den_n == 0:
                continue

            pearson = num / (den_t * den_n)
            if pearson > 0:
                user = self.user_repo.get_by_id(neighbor_id)
                similarities.append(UserSimilarity(
                    user_id=neighbor_id,
                    username=user.username if user else neighbor_id[:8],
                    similarity=pearson,
                    common_items=len(common)
                ))

        # Top-K komşu
        top_k = sorted(similarities, key=lambda x: x.similarity, reverse=True)[
            :self.cfg.cf_top_k_neighbors
        ]
        if not top_k:
            return self.cold_start(limit=limit)

        logger.info(f"[CF] {len(top_k)} komşu bulundu")

        # Ağırlıklı tahmin skoru
        seen_posts = set(target_vector.keys())
        predicted: dict[str, tuple[float, int]] = {}  # post_id → (weighted_sum, voter_count)

        for sim_user in top_k:
            n_vec = neighbor_vectors[sim_user.user_id]
            for post_id, score in n_vec.items():
                if post_id in seen_posts:
                    continue
                ws, vc = predicted.get(post_id, (0.0, 0))
                predicted[post_id] = (ws + sim_user.similarity * score, vc + 1)

        # Güvenilirlik eşiği: en az N komşu oy vermiş olmalı
        filtered = {
            pid: ws / sum(abs(s.similarity) for s in top_k)
            for pid, (ws, vc) in predicted.items()
            if vc >= self.cfg.cf_min_neighbor_votes
        }

        # Normalize ve sırala
        normalized = self._minmax(filtered)
        top_posts = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:limit]

        results = []
        for post_id, score in top_posts:
            post = self.post_repo.get_by_id(post_id)
            if not post:
                continue
            results.append(RecommendedPost(
                post_id=post_id,
                author=post.author.username if post.author else "?",
                content_preview=post.content,
                media_type=post.media_type or "text",
                score=round(score, 4),
                source="cf",
                reason="Takip ettiğin kişiler bunu beğendi"
            ))

        logger.info(f"[CF] {len(results)} öneri üretildi")
        return results

    # ════════════════════════════════════════════════════════
    # 2. CONTENT-BASED FILTERING
    # ════════════════════════════════════════════════════════
    def content_based(self, user_id: str, limit: int = 20) -> list[RecommendedPost]:
        """Hashtag affinitesi + virality + tazelik ile içerik tabanlı öneri."""
        logger.info(f"[CBF] content_based başlatıldı — user={user_id[:8]}")

        # Kullanıcının favori hashtagleri
        rows = self.session.execute(text("""
            SELECT ph.hashtag_id, COUNT(*) AS freq
            FROM post_hashtags ph
            JOIN posts p ON p.id = ph.post_id
            WHERE p.user_id = :uid
              AND p.created_at >= NOW() - INTERVAL '90 days'
            GROUP BY ph.hashtag_id
            ORDER BY freq DESC
            LIMIT 20
        """), {"uid": user_id}).mappings().all()

        user_hashtag_freq: dict[str, int] = {
            r["hashtag_id"]: r["freq"] for r in rows
        }
        max_freq = max(user_hashtag_freq.values()) if user_hashtag_freq else 1

        # Görülmemiş postları getir
        candidate_posts = self.post_repo.get_unseen_by_user(user_id, limit=200)

        scored: list[tuple[str, float, datetime, str]] = []
        for post in candidate_posts:
            # Hashtag affinite skoru
            post_hashtag_ids = {ph.hashtag_id for ph in post.post_hashtags}
            hashtag_score = sum(
                user_hashtag_freq.get(hid, 0) / max_freq
                for hid in post_hashtag_ids
            )
            hashtag_score = min(hashtag_score, 1.0)

            # Tazelik
            freshness = self._freshness(post.created_at)

            # Virality (normalize — 0-100 arası olabilir)
            virality = min(post.virality_score / 100, 1.0)

            # Bileşik skor
            score = (
                hashtag_score * 0.40
                + freshness   * 0.30
                + virality    * 0.30
            )
            scored.append((post.id, score, post.created_at, post.media_type or "text"))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:limit]

        results = []
        for post_id, score, _, media_type in top:
            post = self.post_repo.get_by_id(post_id)
            if not post:
                continue
            results.append(RecommendedPost(
                post_id=post_id,
                author=post.author.username if post.author else "?",
                content_preview=post.content,
                media_type=media_type,
                score=round(score, 4),
                source="cbf",
                reason="İlgi alanlarınla eşleşiyor"
            ))

        logger.info(f"[CBF] {len(results)} öneri üretildi")
        return results

    # ════════════════════════════════════════════════════════
    # 3. HYBRID
    # ════════════════════════════════════════════════════════
    def hybrid(self, user_id: str, limit: int = 20) -> list[RecommendedPost]:
        """CF + CBF + Popularity ağırlıklı hibrit öneri."""
        logger.info(f"[HYBRID] başlatıldı — user={user_id[:8]}")

        cf_recs  = self.user_cf(user_id, limit=limit * 2)
        cbf_recs = self.content_based(user_id, limit=limit * 2)

        cf_scores  = self._minmax({r.post_id: r.score for r in cf_recs})
        cbf_scores = self._minmax({r.post_id: r.score for r in cbf_recs})

        all_posts = set(cf_scores) | set(cbf_scores)

        # Popularity: virality skoru
        pop_scores: dict[str, float] = {}
        for pid in all_posts:
            post = self.post_repo.get_by_id(pid)
            if post:
                pop_scores[pid] = min(post.virality_score / 100, 1.0)
        pop_norm = self._minmax(pop_scores)

        # Kullanıcı cold-start mı?
        is_cold = len(cf_recs) == 0
        cf_w  = 0.30 if is_cold else self.cfg.hybrid_cf_weight
        cbf_w = 0.55 if is_cold else self.cfg.hybrid_cbf_weight
        pop_w = self.cfg.hybrid_popularity_weight

        hybrid_scores: dict[str, float] = {}
        for pid in all_posts:
            hybrid_scores[pid] = (
                cf_scores.get(pid,  0.0) * cf_w
                + cbf_scores.get(pid, 0.0) * cbf_w
                + pop_norm.get(pid,  0.0) * pop_w
            )

        top = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        # Kaynak belirle
        cf_post_ids = {r.post_id for r in cf_recs}
        cbf_post_ids = {r.post_id for r in cbf_recs}

        results = []
        for post_id, score in top:
            post = self.post_repo.get_by_id(post_id)
            if not post:
                continue
            if post_id in cf_post_ids and post_id in cbf_post_ids:
                source, reason = "hybrid", "Beğenilerin ve ilgilerin birleşti"
            elif post_id in cf_post_ids:
                source, reason = "cf",     "Takip ettiğin kişiler beğendi"
            else:
                source, reason = "cbf",    "İlgi alanlarınla eşleşiyor"

            results.append(RecommendedPost(
                post_id=post_id,
                author=post.author.username if post.author else "?",
                content_preview=post.content,
                media_type=post.media_type or "text",
                score=round(score, 4),
                source=source,
                reason=reason
            ))

        logger.info(f"[HYBRID] {len(results)} öneri üretildi (cold_start={is_cold})")
        return results

    # ════════════════════════════════════════════════════════
    # 4. COLD START
    # ════════════════════════════════════════════════════════
    def cold_start(self, limit: int = 20) -> list[RecommendedPost]:
        """
        Yeni kullanıcı için çeşitli ve popüler içerikler.
        Her medya tipinden en iyi N postu getir (diversity zorlanmış).
        """
        logger.info("[COLD_START] cold_start stratejisi uygulanıyor")

        cutoff = datetime.utcnow() - timedelta(hours=self.cfg.cold_start_trending_hours)
        rows = self.session.execute(text("""
            WITH ranked AS (
                SELECT
                    p.id           AS post_id,
                    p.user_id,
                    p.content,
                    p.media_type,
                    p.virality_score,
                    COUNT(i.id)    AS recent_interactions,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.media_type
                        ORDER BY p.virality_score DESC
                    ) AS rn
                FROM posts p
                LEFT JOIN interactions i
                    ON i.post_id = p.id
                    AND i.created_at >= :cutoff
                WHERE p.created_at >= :cutoff
                GROUP BY p.id, p.user_id, p.content, p.media_type, p.virality_score
            )
            SELECT
                r.post_id, r.content, r.media_type,
                r.virality_score, r.recent_interactions,
                u.username,
                r.virality_score * 0.6 + r.recent_interactions * 0.4 AS cold_score
            FROM ranked r
            JOIN users u ON u.id = r.user_id
            WHERE r.rn <= :per_type
            ORDER BY cold_score DESC
            LIMIT :lim
        """), {
            "cutoff": cutoff,
            "per_type": self.cfg.cold_start_per_media_type,
            "lim": limit
        }).mappings().all()

        results = []
        for row in rows:
            results.append(RecommendedPost(
                post_id=row["post_id"],
                author=row["username"],
                content_preview=row["content"],
                media_type=row["media_type"] or "text",
                score=round(float(row["cold_score"]), 4),
                source="cold_start",
                reason="Şu an trend olan içerikler"
            ))

        logger.info(f"[COLD_START] {len(results)} öneri üretildi")
        return results

    # ════════════════════════════════════════════════════════
    # 5. ÖNERİLERİ KAYDET & BATCH PIPELINE
    # ════════════════════════════════════════════════════════
    def persist_recommendations(
        self, user_id: str, recs: list[RecommendedPost],
        model_version: str = "v1.0"
    ) -> int:
        """Üretilen önerileri recommendations tablosuna yaz."""
        records = [
            {
                "source_user_id": user_id,
                "target_post_id": r.post_id,
                "score": r.score,
                "model_version": model_version,
                "was_clicked": False,
            }
            for r in recs
        ]
        self.rec_repo.bulk_save(records)
        logger.info(f"[PERSIST] {len(records)} öneri kaydedildi — user={user_id[:8]}")
        return len(records)

    def run_batch_pipeline(self, model_version: str = "v1.0") -> dict:
        """
        Tüm aktif kullanıcılar için öneri batch'i çalıştırır.
        Scheduler (Celery / APScheduler) tarafından tetiklenir.
        """
        logger.info("[BATCH] Pipeline başlatıldı")

        # 1) Stale önerileri temizle
        deleted = self.rec_repo.delete_stale(self.cfg.recommendation_ttl_days)
        logger.info(f"[BATCH] {deleted} eski öneri silindi")

        # 2) Aktif kullanıcıları getir
        active_users = self.user_repo.get_active_users(since_days=3)
        logger.info(f"[BATCH] {len(active_users)} aktif kullanıcı işlenecek")

        stats = {"processed": 0, "total_recs": 0, "errors": 0}

        for user in active_users:
            try:
                recs = self.hybrid(user.id, limit=self.cfg.daily_recommendation_quota)
                saved = self.persist_recommendations(user.id, recs, model_version)
                stats["processed"] += 1
                stats["total_recs"] += saved
            except Exception as e:
                logger.error(f"[BATCH] Hata — user={user.id[:8]}: {e}")
                stats["errors"] += 1

        logger.info(f"[BATCH] Tamamlandı: {stats}")
        return stats
