import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean,
    ForeignKey, CheckConstraint, UniqueConstraint,
    DateTime, Index, func
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from ..core.database import Base


# ── Yardımcı fonksiyon ──────────────────────────────────────
def new_uuid():
    return str(uuid.uuid4())


# ════════════════════════════════════════════════════════════
# USER
# ════════════════════════════════════════════════════════════
class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=new_uuid
    )
    username: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    bio: Mapped[Optional[str]] = mapped_column(Text)
    location: Mapped[Optional[str]] = mapped_column(String(100))

    follower_count: Mapped[int] = mapped_column(Integer, default=0)
    following_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_engagement_rate: Mapped[float] = mapped_column(Float, default=0.0)

    # pgvector — 384 boyutlu kullanıcı gömme vektörü
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(384))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # İlişkiler
    posts: Mapped[List["Post"]] = relationship(
        "Post", back_populates="author", cascade="all, delete-orphan"
    )
    interactions: Mapped[List["Interaction"]] = relationship(
        "Interaction", back_populates="user", cascade="all, delete-orphan"
    )
    sessions: Mapped[List["UserSession"]] = relationship(
        "UserSession", back_populates="user", cascade="all, delete-orphan"
    )
    recommendations_received: Mapped[List["Recommendation"]] = relationship(
        "Recommendation", back_populates="source_user",
        foreign_keys="Recommendation.source_user_id",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_users_username", "username"),
        Index("idx_users_last_active", "last_active_at"),
    )

    def __repr__(self):
        return f"<User {self.username}>"

    @property
    def follow_ratio(self) -> float:
        return (
            self.follower_count / self.following_count
            if self.following_count > 0 else 0.0
        )


# ════════════════════════════════════════════════════════════
# POST
# ════════════════════════════════════════════════════════════
class Post(Base):
    __tablename__ = "posts"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=new_uuid
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    media_type: Mapped[Optional[str]] = mapped_column(String(20))

    view_count: Mapped[int] = mapped_column(Integer, default=0)
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    comment_count: Mapped[int] = mapped_column(Integer, default=0)
    share_count: Mapped[int] = mapped_column(Integer, default=0)

    virality_score: Mapped[float] = mapped_column(Float, default=0.0)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)

    # pgvector — 384 boyutlu içerik gömme vektörü
    content_embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(384))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # İlişkiler
    author: Mapped["User"] = relationship("User", back_populates="posts")
    interactions: Mapped[List["Interaction"]] = relationship(
        "Interaction", back_populates="post", cascade="all, delete-orphan"
    )
    post_hashtags: Mapped[List["PostHashtag"]] = relationship(
        "PostHashtag", back_populates="post", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "media_type IN ('text','image','video','reel','story')",
            name="chk_media_type"
        ),
        Index("idx_posts_user_created", "user_id", "created_at"),
        Index("idx_posts_virality", "virality_score"),
    )

    def __repr__(self):
        return f"<Post {self.id[:8]}... by {self.user_id[:8]}>"

    def compute_virality(self) -> float:
        """Virality skoru = ağırlıklı engagement / görüntülenme."""
        if self.view_count == 0:
            return 0.0
        weighted = self.like_count + self.comment_count * 2 + self.share_count * 3
        return round(weighted / self.view_count * 100, 4)


# ════════════════════════════════════════════════════════════
# FOLLOWS
# ════════════════════════════════════════════════════════════
class Follow(Base):
    __tablename__ = "follows"

    follower_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True
    )
    following_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True
    )
    interaction_weight: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("follower_id <> following_id", name="chk_no_self_follow"),
        Index("idx_follows_follower",  "follower_id"),
        Index("idx_follows_following", "following_id"),
    )

    def __repr__(self):
        return f"<Follow {self.follower_id[:8]}→{self.following_id[:8]}>"


# ════════════════════════════════════════════════════════════
# INTERACTION
# ════════════════════════════════════════════════════════════
class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=new_uuid
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    post_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("posts.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[str] = mapped_column(String(20), nullable=False)
    dwell_time_ms: Mapped[int] = mapped_column(Integer, default=0)
    scroll_depth: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="interactions")
    post: Mapped["Post"] = relationship("Post", back_populates="interactions")

    __table_args__ = (
        CheckConstraint(
            "type IN ('like','comment','share','view','save','report')",
            name="chk_interaction_type"
        ),
        Index("idx_interactions_user_post", "user_id", "post_id", "type"),
        Index("idx_interactions_post_type", "post_id", "type", "created_at"),
    )

    def __repr__(self):
        return f"<Interaction {self.type} by {self.user_id[:8]}>"

    @property
    def engagement_score(self) -> float:
        """Bu etkileşimin ham sinyal ağırlığı."""
        weights = {
            "share": 5.0, "save": 4.0, "like": 3.0,
            "comment": 2.5, "view": 1.0, "report": -5.0
        }
        base = weights.get(self.type, 0.0)
        if self.type == "view":
            base += min(self.dwell_time_ms / 30_000, 1.0)
        return base


# ════════════════════════════════════════════════════════════
# HASHTAG & POST_HASHTAG
# ════════════════════════════════════════════════════════════
class Hashtag(Base):
    __tablename__ = "hashtags"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=new_uuid
    )
    tag: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    trend_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    post_hashtags: Mapped[List["PostHashtag"]] = relationship(
        "PostHashtag", back_populates="hashtag"
    )

    __table_args__ = (
        Index("idx_hashtags_trend", "trend_score"),
    )

    def __repr__(self):
        return f"<Hashtag #{self.tag}>"


class PostHashtag(Base):
    __tablename__ = "post_hashtags"

    post_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("posts.id", ondelete="CASCADE"),
        primary_key=True
    )
    hashtag_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("hashtags.id", ondelete="CASCADE"),
        primary_key=True
    )

    post: Mapped["Post"] = relationship("Post", back_populates="post_hashtags")
    hashtag: Mapped["Hashtag"] = relationship("Hashtag", back_populates="post_hashtags")


# ════════════════════════════════════════════════════════════
# USER SESSION
# ════════════════════════════════════════════════════════════
class UserSession(Base):
    __tablename__ = "user_sessions"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=new_uuid
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    session_duration_s: Mapped[int] = mapped_column(Integer, default=0)
    posts_viewed: Mapped[int] = mapped_column(Integer, default=0)
    actions_taken: Mapped[int] = mapped_column(Integer, default=0)
    device_type: Mapped[Optional[str]] = mapped_column(String(20))
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="sessions")

    __table_args__ = (
        CheckConstraint(
            "device_type IN ('ios','android','web','desktop')",
            name="chk_device_type"
        ),
        Index("idx_sessions_user_time", "user_id", "started_at"),
    )


# ════════════════════════════════════════════════════════════
# RECOMMENDATION
# ════════════════════════════════════════════════════════════
class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=new_uuid
    )
    source_user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    target_post_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("posts.id", ondelete="CASCADE"), nullable=False
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), default="v1.0")
    was_clicked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    source_user: Mapped["User"] = relationship(
        "User", back_populates="recommendations_received",
        foreign_keys=[source_user_id]
    )
    target_post: Mapped["Post"] = relationship("Post")

    __table_args__ = (
        Index("idx_recs_user_score",    "source_user_id", "score"),
        Index("idx_recs_model_version", "model_version",  "was_clicked"),
        Index("idx_recs_created",       "created_at"),
    )

    def __repr__(self):
        return f"<Recommendation score={self.score:.3f} v={self.model_version}>"
