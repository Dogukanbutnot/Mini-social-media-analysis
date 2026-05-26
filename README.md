"""
Social Media Analytics Backend — Demo Runner
=============================================
Tüm servisleri mock veriyle çalıştırır ve çıktıları gösterir.
Gerçek PostgreSQL bağlantısı olmadan da mantığı inceleyebilirsiniz.
"""

import json
import logging
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import random
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ════════════════════════════════════════════════════════════
# MOCK VERİ KATMANI
# Gerçek DB olmadan servislerin mantığını göstermek için
# ════════════════════════════════════════════════════════════

@dataclass
class MockUser:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    follower_count: int = 0
    following_count: int = 0
    avg_engagement_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class MockPost:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    content: str = ""
    media_type: str = "text"
    like_count: int = 0
    view_count: int = 0
    comment_count: int = 0
    share_count: int = 0
    virality_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class MockInteraction:
    user_id: str
    post_id: str
    type: str
    dwell_time_ms: int = 0
    scroll_depth: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)


def generate_mock_data():
    """Gerçekçi mock veri seti oluştur."""
    random.seed(42)
    users, posts, interactions, follows = [], [], [], []

    usernames = [
        "ali_ml", "zeynep_data", "can_engineer", "selin_ai",
        "mert_dev", "ayse_nlp", "burak_cv", "elif_rec"
    ]

    for uname in usernames:
        user = MockUser(
            username=uname,
            follower_count=random.randint(100, 10_000),
            following_count=random.randint(50, 500),
            avg_engagement_rate=random.uniform(1.0, 8.0),
            created_at=datetime.utcnow() - timedelta(days=random.randint(10, 365)),
            last_active_at=datetime.utcnow() - timedelta(hours=random.randint(0, 120)),
        )
        users.append(user)

    topics = [
        ("Python ile veri analizi nasıl yapılır?", "text"),
        ("Transformer mimarisi görselleştirmesi", "image"),
        ("SQL window functions detaylı anlatım", "text"),
        ("PyTorch ile özel loss fonksiyonu", "text"),
        ("Vector database karşılaştırması: Pinecone vs Weaviate", "text"),
        ("RAG pipeline kurulumu — adım adım", "video"),
        ("LLM fine-tuning best practices", "text"),
        ("FastAPI + SQLAlchemy production setup", "text"),
        ("A/B testing istatistiksel anlamlılık hesabı", "text"),
        ("Recommendation system mimari tasarımı", "image"),
        ("Embedding modelleri benchmark sonuçları", "text"),
        ("Kafka ile real-time ML pipeline", "video"),
    ]

    for user in users:
        for content, media_type in random.sample(topics, k=random.randint(2, 5)):
            view_count = random.randint(200, 15_000)
            like_count = int(view_count * random.uniform(0.02, 0.12))
            comment_count = int(like_count * random.uniform(0.1, 0.4))
            share_count = int(like_count * random.uniform(0.05, 0.2))
            virality = (like_count + comment_count * 2 + share_count * 3) / view_count * 100

            post = MockPost(
                user_id=user.id,
                content=content,
                media_type=media_type,
                like_count=like_count,
                view_count=view_count,
                comment_count=comment_count,
                share_count=share_count,
                virality_score=round(virality, 3),
                created_at=datetime.utcnow() - timedelta(hours=random.randint(0, 168)),
            )
            posts.append(post)

    # Etkileşimler
    interaction_types = ["like", "view", "save", "share", "comment"]
    for user in users:
        sample_posts = random.sample(posts, k=min(15, len(posts)))
        for post in sample_posts:
            if post.user_id == user.id:
                continue
            itype = random.choices(
                interaction_types, weights=[25, 50, 10, 8, 7], k=1
            )[0]
            interactions.append(MockInteraction(
                user_id=user.id,
                post_id=post.id,
                type=itype,
                dwell_time_ms=random.randint(1_000, 120_000) if itype == "view" else 0,
                scroll_depth=random.uniform(0.1, 1.0),
                created_at=datetime.utcnow() - timedelta(hours=random.randint(0, 720)),
            ))

    # Takip ilişkileri
    for i, user in enumerate(users):
        targets = random.sample([u for u in users if u.id != user.id], k=random.randint(2, 5))
        for target in targets:
            follows.append((user.id, target.id, random.uniform(0.1, 1.0)))

    return users, posts, interactions, follows


# ════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE (mock bağımsız)
# ════════════════════════════════════════════════════════════

class MockRecommendationEngine:
    """
    Gerçek DB olmadan çalışan recommendation engine demo.
    Aynı algoritmaları kullanır, veri mock'tan gelir.
    """

    def __init__(self, users, posts, interactions, follows):
        self.users = {u.id: u for u in users}
        self.posts = {p.id: p for p in posts}
        self.interactions = interactions
        self.follows = follows  # [(follower_id, following_id, weight)]

        # {user_id: {post_id: score}}
        self.user_vectors = self._build_user_vectors()

    def _signal_weight(self, itype: str, dwell_ms: int = 0) -> float:
        weights = {"share": 5.0, "save": 4.0, "like": 3.0,
                   "comment": 2.5, "view": 1.0, "report": -5.0}
        base = weights.get(itype, 0.0)
        if itype == "view":
            base += min(dwell_ms / 30_000, 1.0)
        return base

    def _build_user_vectors(self) -> dict:
        vectors = {}
        for inter in self.interactions:
            uid, pid = inter.user_id, inter.post_id
            score = self._signal_weight(inter.type, inter.dwell_time_ms)
            vectors.setdefault(uid, {})[pid] = vectors.get(uid, {}).get(pid, 0) + score
        return vectors

    def _minmax(self, scores: dict) -> dict:
        if not scores:
            return {}
        lo, hi = min(scores.values()), max(scores.values())
        return {k: (v - lo) / (hi - lo) if hi != lo else 1.0 for k, v in scores.items()}

    def _freshness(self, created_at: datetime, half_life_hours: float = 24.0) -> float:
        hours_old = (datetime.utcnow() - created_at).total_seconds() / 3600
        return math.exp(-0.693 * hours_old / half_life_hours)

    def user_cf(self, user_id: str, limit: int = 5) -> list[dict]:
        """Pearson korelasyonu ile CF önerileri."""
        target_vec = self.user_vectors.get(user_id, {})
        if not target_vec:
            return self.cold_start(limit=limit)

        target_mean = sum(target_vec.values()) / len(target_vec)
        following_ids = {fid for (fol, fid, _) in self.follows if fol == user_id}

        similarities = []
        for uid, n_vec in self.user_vectors.items():
            if uid == user_id or uid not in following_ids:
                continue
            common = set(target_vec) & set(n_vec)
            if len(common) < 3:
                continue
            n_mean = sum(n_vec.values()) / len(n_vec)
            num = sum((target_vec[p] - target_mean) * (n_vec[p] - n_mean) for p in common)
            den_t = math.sqrt(sum((target_vec[p] - target_mean) ** 2 for p in common))
            den_n = math.sqrt(sum((n_vec[p] - n_mean) ** 2 for p in common))
            if den_t * den_n == 0:
                continue
            pearson = num / (den_t * den_n)
            if pearson > 0:
                similarities.append((uid, pearson, n_vec))

        top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
        if not top_k:
            return self.cold_start(limit=limit)

        seen = set(target_vec.keys())
        predicted = {}
        for uid, sim, n_vec in top_k:
            for pid, score in n_vec.items():
                if pid in seen:
                    continue
                ws, vc = predicted.get(pid, (0.0, 0))
                predicted[pid] = (ws + sim * score, vc + 1)

        filtered = {
            pid: ws / sum(abs(s) for _, s, _ in top_k)
            for pid, (ws, vc) in predicted.items()
            if vc >= 2
        }
        normalized = self._minmax(filtered)
        top = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:limit]

        results = []
        for pid, score in top:
            post = self.posts.get(pid)
            author = self.users.get(post.user_id) if post else None
            if post:
                results.append({
                    "post_id": pid[:8],
                    "author": author.username if author else "?",
                    "content": post.content[:60],
                    "score": round(score, 4),
                    "source": "user_cf",
                    "reason": "Takip ettiğin kişiler beğendi",
                })
        return results

    def content_based(self, user_id: str, limit: int = 5) -> list[dict]:
        """Hashtag affinitesi + virality + tazelik ile CBF."""
        seen_ids = set(self.user_vectors.get(user_id, {}).keys())
        candidates = [p for p in self.posts.values() if p.id not in seen_ids
                      and p.user_id != user_id]

        scored = []
        for post in candidates:
            freshness = self._freshness(post.created_at)
            virality = min(post.virality_score / 100, 1.0)
            score = freshness * 0.40 + virality * 0.60
            scored.append((post, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for post, score in scored[:limit]:
            author = self.users.get(post.user_id)
            results.append({
                "post_id": post.id[:8],
                "author": author.username if author else "?",
                "content": post.content[:60],
                "score": round(score, 4),
                "source": "content_based",
                "reason": "İlgi alanlarınla eşleşiyor",
            })
        return results

    def hybrid(self, user_id: str, limit: int = 5) -> list[dict]:
        """CF + CBF + Popularity hibrit öneri."""
        cf  = self.user_cf(user_id, limit=limit * 2)
        cbf = self.content_based(user_id, limit=limit * 2)

        cf_scores  = self._minmax({r["post_id"]: r["score"] for r in cf})
        cbf_scores = self._minmax({r["post_id"]: r["score"] for r in cbf})

        all_ids = set(cf_scores) | set(cbf_scores)
        pop_scores = {
            pid: min(self.posts.get(pid + "...", MockPost()).virality_score / 100, 1.0)
            for pid in all_ids
        }

        hybrid = {
            pid: (cf_scores.get(pid, 0) * 0.50
                  + cbf_scores.get(pid, 0) * 0.35
                  + pop_scores.get(pid, 0) * 0.15)
            for pid in all_ids
        }

        top = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:limit]
        cf_ids  = {r["post_id"] for r in cf}
        cbf_ids = {r["post_id"] for r in cbf}

        results = []
        for pid, score in top:
            if pid in cf_ids and pid in cbf_ids:
                source, reason = "hybrid", "Beğenilerin ve ilgilerin birleşti"
            elif pid in cf_ids:
                source, reason = "cf", "Takip ettiğin kişiler beğendi"
            else:
                source, reason = "cbf", "İlgi alanlarınla eşleşiyor"

            # İlgili postu bul (kısa ID ile eşleşen)
            full_post = next((p for p in self.posts.values() if p.id.startswith(pid)), None)
            if not full_post:
                continue
            author = self.users.get(full_post.user_id)
            results.append({
                "post_id": pid,
                "author": author.username if author else "?",
                "content": full_post.content[:60],
                "score": round(score, 4),
                "source": source,
                "reason": reason,
            })
        return results

    def cold_start(self, limit: int = 5) -> list[dict]:
        """Yeni kullanıcı için trending içerikler."""
        cutoff = datetime.utcnow() - timedelta(hours=72)
        recent = [p for p in self.posts.values() if p.created_at >= cutoff]

        # Medya tipi başına en iyi 2
        by_type: dict[str, list] = {}
        for p in recent:
            by_type.setdefault(p.media_type, []).append(p)
        for mt in by_type:
            by_type[mt].sort(key=lambda p: p.virality_score, reverse=True)

        diverse = []
        for mt, ps in by_type.items():
            diverse.extend(ps[:2])
        diverse.sort(key=lambda p: p.virality_score, reverse=True)

        results = []
        for post in diverse[:limit]:
            author = self.users.get(post.user_id)
            results.append({
                "post_id": post.id[:8],
                "author": author.username if author else "?",
                "content": post.content[:60],
                "score": round(min(post.virality_score / 100, 1.0), 4),
                "source": "cold_start",
                "reason": "Şu an trend olan içerikler",
            })
        return results


# ════════════════════════════════════════════════════════════
# ANALYTICS (mock)
# ════════════════════════════════════════════════════════════

class MockAnalyticsService:
    def __init__(self, users, posts, interactions):
        self.users = {u.id: u for u in users}
        self.posts = {p.id: p for p in posts}
        self.interactions = interactions

    def engagement_summary(self) -> list[dict]:
        """Kullanıcı başına engagement özeti."""
        from collections import defaultdict
        stats: dict[str, dict] = defaultdict(lambda: {
            "likes": 0, "views": 0, "shares": 0, "comments": 0, "dwell_total": 0, "count": 0
        })
        post_to_user = {p.id: p.user_id for p in self.posts.values()}

        for inter in self.interactions:
            uid = post_to_user.get(inter.post_id)
            if not uid:
                continue
            s = stats[uid]
            s[inter.type + "s"] = s.get(inter.type + "s", 0) + 1
            if inter.type == "view":
                s["views"] += 1
                s["dwell_total"] += inter.dwell_time_ms
                s["count"] += 1

        results = []
        for uid, s in stats.items():
            user = self.users.get(uid)
            if not user:
                continue
            er = (s["likes"] + s.get("comments", 0) * 2 + s.get("shares", 0) * 3) / max(s["views"], 1) * 100
            results.append({
                "user_id": uid[:8],
                "username": user.username,
                "total_views": s["views"],
                "total_likes": s["likes"],
                "total_shares": s.get("shares", 0),
                "engagement_rate_pct": round(er, 2),
                "avg_dwell_ms": round(s["dwell_total"] / max(s["count"], 1), 0),
            })
        return sorted(results, key=lambda x: x["engagement_rate_pct"], reverse=True)

    def churn_analysis(self) -> list[dict]:
        """Basit churn analizi."""
        results = []
        now = datetime.utcnow()
        for user in self.users.values():
            days_inactive = (now - user.last_active_at).days
            if days_inactive == 0:
                label = "healthy"
                risk = 0.05
            elif days_inactive <= 2:
                label = "declining"
                risk = 0.3
            elif days_inactive <= 4:
                label = "at_risk"
                risk = 0.65
            else:
                label = "churned"
                risk = 0.9
            results.append({
                "username": user.username,
                "days_inactive": days_inactive,
                "churn_risk": round(risk, 3),
                "label": label,
            })
        return sorted(results, key=lambda x: x["churn_risk"], reverse=True)

    def influence_scores(self, follows) -> list[dict]:
        """Basit etki skoru hesabı."""
        direct: dict[str, float] = {}
        for fol, fid, weight in follows:
            direct[fid] = direct.get(fid, 0) + weight

        results = []
        for uid, score in sorted(direct.items(), key=lambda x: x[1], reverse=True)[:8]:
            user = self.users.get(uid)
            if user:
                results.append({
                    "username": user.username,
                    "influence_score": round(score, 3),
                    "follower_weight_sum": round(score, 3),
                })
        return results


# ════════════════════════════════════════════════════════════
# A/B TEST (mock)
# ════════════════════════════════════════════════════════════

class MockABTestService:
    @staticmethod
    def _z_to_p(z: float) -> float:
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        poly = sum(c * t**i for i, c in enumerate(
            [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429], 1
        ))
        return min(2 * poly * math.exp(-0.5 * z**2) / math.sqrt(2 * math.pi), 1.0)

    def compare_models(self) -> dict:
        """İki model sürümü karşılaştırması (simülasyon)."""
        random.seed(99)
        n_a, c_a = 1200, 144   # v1.0: CTR %12
        n_b, c_b = 1150, 161   # v2.0: CTR %14

        p_a = c_a / n_a
        p_b = c_b / n_b
        p_pool = (c_a + c_b) / (n_a + n_b)
        std_err = math.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        z = (p_b - p_a) / std_err
        p_value = self._z_to_p(z)

        return {
            "version_a": "v1.0",
            "version_b": "v2.0",
            "n_a": n_a, "clicks_a": c_a, "ctr_a": f"{p_a*100:.2f}%",
            "n_b": n_b, "clicks_b": c_b, "ctr_b": f"{p_b*100:.2f}%",
            "ctr_lift": f"{(p_b - p_a)*100:+.2f}%",
            "z_score": round(z, 4),
            "p_value": round(p_value, 4),
            "is_significant": p_value < 0.05,
            "winner": "v2.0" if p_value < 0.05 and p_b > p_a else "Fark anlamlı değil",
        }


# ════════════════════════════════════════════════════════════
# ANA DEMO
# ════════════════════════════════════════════════════════════

def print_section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

def print_results(items: list, max_n: int = 5):
    for i, item in enumerate(items[:max_n], 1):
        if isinstance(item, dict):
            parts = " | ".join(f"{k}={v}" for k, v in item.items())
            print(f"  {i:2d}. {parts}")
        else:
            print(f"  {i:2d}. {item}")


def main():
    logger.info("Mock veri oluşturuluyor...")
    users, posts, interactions, follows = generate_mock_data()
    logger.info(
        f"Oluşturuldu: {len(users)} kullanıcı, "
        f"{len(posts)} post, {len(interactions)} etkileşim, "
        f"{len(follows)} takip"
    )

    target_user = users[0]
    engine = MockRecommendationEngine(users, posts, interactions, follows)
    analytics = MockAnalyticsService(users, posts, interactions)
    ab_test = MockABTestService()

    # ── 1. Cold Start ──────────────────────────────────────
    print_section("1. COLD START — Yeni Kullanıcı Önerileri")
    cold = engine.cold_start(limit=4)
    print_results(cold)

    # ── 2. User-Based CF ──────────────────────────────────
    print_section(f"2. USER-BASED CF — '{target_user.username}' için")
    cf_recs = engine.user_cf(target_user.id, limit=4)
    if cf_recs:
        print_results(cf_recs)
    else:
        print("  (Yeterli komşu bulunamadı — cold start uygulandı)")

    # ── 3. Content-Based ──────────────────────────────────
    print_section(f"3. CONTENT-BASED — '{target_user.username}' için")
    cbf_recs = engine.content_based(target_user.id, limit=4)
    print_results(cbf_recs)

    # ── 4. Hybrid ─────────────────────────────────────────
    print_section(f"4. HYBRID — '{target_user.username}' için")
    hybrid_recs = engine.hybrid(target_user.id, limit=5)
    print_results(hybrid_recs)

    # ── 5. Engagement Analizi ─────────────────────────────
    print_section("5. ENGAGEMENT ANALİZİ — Top Kullanıcılar")
    eng = analytics.engagement_summary()
    print_results(eng, max_n=5)

    # ── 6. Churn Analizi ──────────────────────────────────
    print_section("6. CHURN ANALİZİ")
    churn = analytics.churn_analysis()
    emoji_map = {"churned": "💀", "at_risk": "⚠️", "declining": "📉", "healthy": "✅"}
    for c in churn:
        e = emoji_map.get(c["label"], "?")
        print(f"  {e} {c['username']:20s} risk={c['churn_risk']:.3f} | "
              f"inactive={c['days_inactive']}d | [{c['label']}]")

    # ── 7. Network Etkisi ─────────────────────────────────
    print_section("7. NETWORK ETKİ SKORLARI")
    influence = analytics.influence_scores(follows)
    for i, rec in enumerate(influence, 1):
        print(f"  #{i:2d} {rec['username']:20s} influence={rec['influence_score']:.3f}")

    # ── 8. A/B Test ───────────────────────────────────────
    print_section("8. A/B TEST — Model Karşılaştırması")
    ab = ab_test.compare_models()
    print(f"  v1.0: {ab['n_a']} öneri, {ab['clicks_a']} tıklama → CTR {ab['ctr_a']}")
    print(f"  v2.0: {ab['n_b']} öneri, {ab['clicks_b']} tıklama → CTR {ab['ctr_b']}")
    print(f"  Lift: {ab['ctr_lift']}  |  z={ab['z_score']}  |  p={ab['p_value']}")
    sig = "✅ İstatistiksel olarak ANLAMLI" if ab["is_significant"] else "⚠️ Anlamlı değil"
    print(f"  {sig}  →  Kazanan: {ab['winner']}")

    print(f"\n{'═'*60}")
    print("  Demo tamamlandı.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
