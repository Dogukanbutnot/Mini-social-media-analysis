from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..models.orm import User, Post, Follow, Interaction, Hashtag, PostHashtag, UserSession, Recommendation


# ════════════════════════════════════════════════════════════
# USER REPOSITORY
# ════════════════════════════════════════════════════════════
class UserRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, user_id: str) -> Optional[User]:
        return self.session.query(User).filter(User.id == user_id).first()

    def get_by_username(self, username: str) -> Optional[User]:
        return self.session.query(User).filter(User.username == username).first()

    def get_active_users(self, since_days: int = 3) -> list[User]:
        cutoff = datetime.utcnow() - timedelta(days=since_days)
        return (
            self.session.query(User)
            .filter(User.last_active_at >= cutoff)
            .order_by(User.last_active_at.desc())
            .all()
        )

    def create(self, username: str, email: str, bio: str = None, location: str = None) -> User:
        user = User(username=username, email=email, bio=bio, location=location)
        self.session.add(user)
        self.session.flush()
        return user

    def update_engagement_rate(self, user_id: str, rate: float):
        self.session.query(User).filter(User.id == user_id).update(
            {"avg_engagement_rate": rate}
        )

    def update_last_active(self, user_id: str):
        self.session.query(User).filter(User.id == user_id).update(
            {"last_active_at": datetime.utcnow()}
        )


# ════════════════════════════════════════════════════════════
# POST REPOSITORY
# ════════════════════════════════════════════════════════════
class PostRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, post_id: str) -> Optional[Post]:
        return self.session.query(Post).filter(Post.id == post_id).first()

    def get_recent(self, limit: int = 50, since_hours: int = 72) -> list[Post]:
        cutoff = datetime.utcnow() - timedelta(hours=since_hours)
        return (
            self.session.query(Post)
            .filter(Post.created_at >= cutoff)
            .order_by(Post.created_at.desc())
            .limit(limit)
            .all()
        )

    def get_top_viral(self, limit: int = 20, since_days: int = 7) -> list[Post]:
        cutoff = datetime.utcnow() - timedelta(days=since_days)
        return (
            self.session.query(Post)
            .filter(Post.created_at >= cutoff)
            .order_by(Post.virality_score.desc())
            .limit(limit)
            .all()
        )

    def get_by_user(self, user_id: str, limit: int = 20) -> list[Post]:
        return (
            self.session.query(Post)
            .filter(Post.user_id == user_id)
            .order_by(Post.created_at.desc())
            .limit(limit)
            .all()
        )

    def create(self, user_id: str, content: str, media_type: str = "text") -> Post:
        post = Post(user_id=user_id, content=content, media_type=media_type)
        self.session.add(post)
        self.session.flush()
        return post

    def update_virality(self, post_id: str, score: float):
        self.session.query(Post).filter(Post.id == post_id).update(
            {"virality_score": score}
        )

    def get_unseen_by_user(self, user_id: str, limit: int = 100) -> list[Post]:
        seen_subq = (
            self.session.query(Interaction.post_id)
            .filter(Interaction.user_id == user_id)
            .subquery()
        )
        return (
            self.session.query(Post)
            .filter(Post.id.notin_(seen_subq))
            .order_by(Post.virality_score.desc())
            .limit(limit)
            .all()
        )


# ════════════════════════════════════════════════════════════
# INTERACTION REPOSITORY
# ════════════════════════════════════════════════════════════
class InteractionRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, user_id: str, post_id: str, type_: str,
               dwell_time_ms: int = 0, scroll_depth: float = 0.0) -> Interaction:
        interaction = Interaction(
            user_id=user_id, post_id=post_id, type=type_,
            dwell_time_ms=dwell_time_ms, scroll_depth=scroll_depth
        )
        self.session.add(interaction)
        self.session.flush()
        return interaction

    def get_user_interactions(self, user_id: str, since_days: int = 30) -> list[Interaction]:
        cutoff = datetime.utcnow() - timedelta(days=since_days)
        return (
            self.session.query(Interaction)
            .filter(Interaction.user_id == user_id, Interaction.created_at >= cutoff)
            .order_by(Interaction.created_at.desc())
            .all()
        )

    def get_post_interactions(self, post_id: str, type_: str = None) -> list[Interaction]:
        q = self.session.query(Interaction).filter(Interaction.post_id == post_id)
        if type_:
            q = q.filter(Interaction.type == type_)
        return q.all()

    def get_user_item_matrix_raw(self) -> list[dict]:
        """Ham user-item etkileşim matrisini döner (ML girdi hazırlığı)."""
        rows = self.session.execute(text("""
            SELECT
                user_id,
                post_id,
                MAX(CASE WHEN type='share'  THEN 5.0 ELSE 0 END) AS share_score,
                MAX(CASE WHEN type='save'   THEN 4.0 ELSE 0 END) AS save_score,
                MAX(CASE WHEN type='like'   THEN 3.0 ELSE 0 END) AS like_score,
                MAX(CASE WHEN type='report' THEN -5.0 ELSE 0 END) AS report_score,
                MAX(CASE WHEN type='view'
                    THEN 1.0 + LEAST(dwell_time_ms / 30000.0, 1.0)
                    ELSE 0 END) AS view_score,
                MAX(scroll_depth) AS max_scroll
            FROM interactions
            GROUP BY user_id, post_id
        """)).mappings().all()
        return [dict(r) for r in rows]


# ════════════════════════════════════════════════════════════
# FOLLOW REPOSITORY
# ════════════════════════════════════════════════════════════
class FollowRepository:
    def __init__(self, session: Session):
        self.session = session

    def follow(self, follower_id: str, following_id: str) -> Follow:
        follow = Follow(follower_id=follower_id, following_id=following_id)
        self.session.add(follow)
        self.session.flush()
        return follow

    def unfollow(self, follower_id: str, following_id: str):
        self.session.query(Follow).filter(
            Follow.follower_id == follower_id,
            Follow.following_id == following_id
        ).delete()

    def get_following_ids(self, user_id: str) -> list[str]:
        rows = (
            self.session.query(Follow.following_id)
            .filter(Follow.follower_id == user_id)
            .all()
        )
        return [r[0] for r in rows]

    def get_follower_ids(self, user_id: str) -> list[str]:
        rows = (
            self.session.query(Follow.follower_id)
            .filter(Follow.following_id == user_id)
            .all()
        )
        return [r[0] for r in rows]

    def update_interaction_weight(self, follower_id: str, following_id: str, weight: float):
        self.session.query(Follow).filter(
            Follow.follower_id == follower_id,
            Follow.following_id == following_id
        ).update({"interaction_weight": weight})


# ════════════════════════════════════════════════════════════
# RECOMMENDATION REPOSITORY
# ════════════════════════════════════════════════════════════
class RecommendationRepository:
    def __init__(self, session: Session):
        self.session = session

    def save(self, user_id: str, post_id: str, score: float,
             model_version: str = "v1.0") -> Recommendation:
        rec = Recommendation(
            source_user_id=user_id, target_post_id=post_id,
            score=score, model_version=model_version
        )
        self.session.add(rec)
        self.session.flush()
        return rec

    def bulk_save(self, records: list[dict]):
        """Toplu öneri kayıt — batch pipeline için."""
        self.session.bulk_insert_mappings(Recommendation, records)

    def mark_clicked(self, rec_id: str):
        self.session.query(Recommendation).filter(
            Recommendation.id == rec_id
        ).update({"was_clicked": True})

    def get_for_user(self, user_id: str, limit: int = 20) -> list[Recommendation]:
        return (
            self.session.query(Recommendation)
            .filter(Recommendation.source_user_id == user_id)
            .order_by(Recommendation.score.desc())
            .limit(limit)
            .all()
        )

    def delete_stale(self, ttl_days: int = 7) -> int:
        cutoff = datetime.utcnow() - timedelta(days=ttl_days)
        deleted = (
            self.session.query(Recommendation)
            .filter(
                Recommendation.created_at < cutoff,
                Recommendation.was_clicked == False
            )
            .delete()
        )
        return deleted

    def get_model_stats(self) -> list[dict]:
        rows = self.session.execute(text("""
            SELECT
                model_version,
                COUNT(*) AS total_recs,
                COUNT(*) FILTER (WHERE was_clicked) AS clicks,
                ROUND(
                    COUNT(*) FILTER (WHERE was_clicked) * 100.0 / NULLIF(COUNT(*),0),
                2) AS ctr_pct,
                AVG(score) AS avg_score
            FROM recommendations
            WHERE created_at >= NOW() - INTERVAL '14 days'
            GROUP BY model_version
            ORDER BY ctr_pct DESC
        """)).mappings().all()
        return [dict(r) for r in rows]

    def get_precision_at_k(self, k: int = 10) -> float:
        row = self.session.execute(text(f"""
            WITH ranked AS (
                SELECT
                    source_user_id,
                    was_clicked,
                    ROW_NUMBER() OVER (
                        PARTITION BY source_user_id ORDER BY score DESC
                    ) AS rk
                FROM recommendations
                WHERE created_at >= NOW() - INTERVAL '7 days'
            )
            SELECT
                AVG(clicked_in_k::FLOAT / {k}) AS precision_at_k
            FROM (
                SELECT
                    source_user_id,
                    COUNT(*) FILTER (WHERE was_clicked AND rk <= {k}) AS clicked_in_k
                FROM ranked
                GROUP BY source_user_id
            ) sub
        """)).first()
        return float(row[0]) if row and row[0] else 0.0
