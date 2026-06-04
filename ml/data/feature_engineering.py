"""
Feature Engineering
===================
PostgreSQL'den ham veriyi çekip ML pipeline'ına hazır
feature matrislerine dönüştürür.

Üretilen çıktılar:
  - user_item_matrix   : scipy sparse (implicit feedback)
  - user_features_df   : pandas DataFrame (kullanıcı özellikleri)
  - post_features_df   : pandas DataFrame (içerik özellikleri)
  - interaction_df     : ham etkileşim logu
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


# ── Sinyal ağırlıkları ────────────────────────────────────────
SIGNAL_WEIGHTS: dict[str, float] = {
    "share":   5.0,
    "save":    4.0,
    "like":    3.0,
    "comment": 2.5,
    "view":    1.0,   # + dwell time bonus
    "report": -5.0,
}


@dataclass
class FeatureSet:
    """Pipeline boyunca taşınan tüm feature'ları bir arada tutar."""
    # Sparse user-item matrisi (ALS / SVD için)
    user_item_matrix: Optional[csr_matrix] = None
    # Matris indeks eşlemeleri
    user_index: dict[str, int] = field(default_factory=dict)   # user_id → satır indeksi
    item_index: dict[str, int] = field(default_factory=dict)   # post_id → sütun indeksi
    index_user: dict[int, str] = field(default_factory=dict)   # satır → user_id
    index_item: dict[int, str] = field(default_factory=dict)   # sütun → post_id
    # DataFrame'ler
    user_features: Optional[pd.DataFrame] = None
    post_features: Optional[pd.DataFrame] = None
    interaction_df: Optional[pd.DataFrame] = None
    # Meta
    n_users: int = 0
    n_items: int = 0
    n_interactions: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


def compute_interaction_score(row: pd.Series) -> float:
    """
    Ham etkileşim satırından bileşik sinyal skoru hesaplar.
    Implicit ALS için confidence değeri olarak kullanılır.
    """
    base = SIGNAL_WEIGHTS.get(row["type"], 0.0)
    if row["type"] == "view":
        # Dwell time bonusu: 30 saniyede +1.0 (maks)
        base += min(row.get("dwell_time_ms", 0) / 30_000, 1.0)
    # Scroll depth bonusu
    base += row.get("scroll_depth", 0.0) * 0.5
    return max(base, 0.0)


def build_user_item_matrix(interaction_df: pd.DataFrame) -> tuple[
    csr_matrix,
    dict[str, int], dict[str, int],
    dict[int, str], dict[int, str]
]:
    """
    Etkileşim DataFrame'inden scipy sparse matris üretir.

    Returns:
        matrix         : (n_users, n_items) csr_matrix — değerler sinyal skoru
        user_index     : user_id → satır indeksi
        item_index     : post_id → sütun indeksi
        index_user     : satır indeksi → user_id
        index_item     : sütun indeksi → post_id
    """
    # Unique id'leri sırala (tekrarlanabilirlik için)
    users = sorted(interaction_df["user_id"].unique())
    items = sorted(interaction_df["post_id"].unique())

    user_index = {uid: i for i, uid in enumerate(users)}
    item_index = {pid: j for j, pid in enumerate(items)}
    index_user = {v: k for k, v in user_index.items()}
    index_item = {v: k for k, v in item_index.items()}

    # Her (user, item) çifti için maksimum skoru al
    agg = (
        interaction_df
        .groupby(["user_id", "post_id"])["interaction_score"]
        .max()
        .reset_index()
    )

    rows = agg["user_id"].map(user_index).values
    cols = agg["post_id"].map(item_index).values
    data = agg["interaction_score"].values.astype(np.float32)

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(users), len(items)),
        dtype=np.float32
    )

    logger.info(
        f"User-item matrisi: {matrix.shape} | "
        f"density={matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}"
    )
    return matrix, user_index, item_index, index_user, index_item


def build_user_features(interaction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Her kullanıcı için ML feature vektörü üretir.

    Kolonlar:
        user_id, total_interactions, unique_posts, like_rate,
        share_rate, save_rate, avg_dwell_ms, avg_scroll,
        session_count (proxy), recency_days, log_interactions
    """
    now = datetime.utcnow()

    grp = interaction_df.groupby("user_id")

    df = pd.DataFrame({
        "user_id": grp["post_id"].count().index,
        "total_interactions": grp["post_id"].count().values,
        "unique_posts":       grp["post_id"].nunique().values,
        "avg_score":          grp["interaction_score"].mean().values,
        "max_score":          grp["interaction_score"].max().values,
    })

    # Tip bazlı oranlar
    for t in ["like", "share", "save", "view", "comment"]:
        counts = interaction_df[interaction_df["type"] == t].groupby("user_id")["post_id"].count()
        df[f"{t}_count"] = df["user_id"].map(counts).fillna(0).astype(int)

    df["like_rate"]    = df["like_count"]    / df["total_interactions"].clip(lower=1)
    df["share_rate"]   = df["share_count"]   / df["total_interactions"].clip(lower=1)
    df["save_rate"]    = df["save_count"]    / df["total_interactions"].clip(lower=1)
    df["comment_rate"] = df["comment_count"] / df["total_interactions"].clip(lower=1)

    # Dwell time & scroll
    dwell = interaction_df[interaction_df["type"] == "view"].groupby("user_id")["dwell_time_ms"].mean()
    scroll = interaction_df.groupby("user_id")["scroll_depth"].mean()
    df["avg_dwell_ms"] = df["user_id"].map(dwell).fillna(0)
    df["avg_scroll"]   = df["user_id"].map(scroll).fillna(0)

    # Recency (son etkileşimden bu yana gün sayısı)
    last_seen = interaction_df.groupby("user_id")["created_at"].max()
    df["last_interaction"] = df["user_id"].map(last_seen)
    df["recency_days"] = (now - df["last_interaction"]).dt.total_seconds() / 86400
    df["recency_days"] = df["recency_days"].fillna(999).clip(upper=365)

    # Log transform (çarpık dağılımlar için)
    df["log_interactions"] = np.log1p(df["total_interactions"])
    df["log_unique_posts"] = np.log1p(df["unique_posts"])

    df = df.drop(columns=["last_interaction"])
    df = df.set_index("user_id")

    logger.info(f"User features: {df.shape} — {df.columns.tolist()}")
    return df


def build_post_features(interaction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Her post için ML feature vektörü üretir.

    Kolonlar:
        post_id, total_interactions, unique_users, popularity_score,
        like_rate, share_rate, avg_dwell_ms, avg_scroll,
        recency_days, log_interactions
    """
    now = datetime.utcnow()

    grp = interaction_df.groupby("post_id")

    df = pd.DataFrame({
        "post_id":            grp["user_id"].count().index,
        "total_interactions": grp["user_id"].count().values,
        "unique_users":       grp["user_id"].nunique().values,
        "avg_score":          grp["interaction_score"].mean().values,
    })

    # Popularity skoru: ağırlıklı etkileşim toplamı / unique kullanıcı
    agg_scores = interaction_df.groupby("post_id")["interaction_score"].sum()
    df["popularity_score"] = df["post_id"].map(agg_scores) / df["unique_users"].clip(lower=1)

    # Tip bazlı oranlar
    for t in ["like", "share", "save", "comment"]:
        counts = interaction_df[interaction_df["type"] == t].groupby("post_id")["user_id"].count()
        df[f"{t}_count"] = df["post_id"].map(counts).fillna(0).astype(int)

    df["like_rate"]    = df["like_count"]    / df["total_interactions"].clip(lower=1)
    df["share_rate"]   = df["share_count"]   / df["total_interactions"].clip(lower=1)
    df["save_rate"]    = df["save_count"]    / df["total_interactions"].clip(lower=1)
    df["comment_rate"] = df["comment_count"] / df["total_interactions"].clip(lower=1)

    # Dwell time & scroll
    dwell  = interaction_df[interaction_df["type"] == "view"].groupby("post_id")["dwell_time_ms"].mean()
    scroll = interaction_df.groupby("post_id")["scroll_depth"].mean()
    df["avg_dwell_ms"] = df["post_id"].map(dwell).fillna(0)
    df["avg_scroll"]   = df["post_id"].map(scroll).fillna(0)

    # Recency
    first_seen = interaction_df.groupby("post_id")["created_at"].min()
    df["post_age_days"] = (now - df["post_id"].map(first_seen)).dt.total_seconds() / 86400
    df["post_age_days"] = df["post_age_days"].fillna(0).clip(upper=365)

    # Tazelik (exponential decay, 24 saatlik yarı-ömür)
    df["freshness"] = np.exp(-0.693 * df["post_age_days"] * 24 / 24)

    # Log transform
    df["log_interactions"] = np.log1p(df["total_interactions"])
    df["log_unique_users"] = np.log1p(df["unique_users"])

    df = df.set_index("post_id")
    logger.info(f"Post features: {df.shape} — {df.columns.tolist()}")
    return df


def prepare_features(interaction_records: list[dict]) -> FeatureSet:
    """
    Ham etkileşim kayıtlarından tam FeatureSet üretir.

    Args:
        interaction_records: [{"user_id", "post_id", "type",
                                "dwell_time_ms", "scroll_depth", "created_at"}, ...]
    Returns:
        FeatureSet: pipeline'a hazır veri paketi
    """
    logger.info(f"[FEATURE] {len(interaction_records)} etkileşim işleniyor")

    df = pd.DataFrame(interaction_records)
    df["created_at"] = pd.to_datetime(df["created_at"])

    # Negatif sinyal filtresi — report'ları düşür (isteğe bağlı)
    df = df[df["type"] != "report"].copy()

    # Sinyal skoru hesapla
    df["interaction_score"] = df.apply(compute_interaction_score, axis=1)
    df = df[df["interaction_score"] > 0].copy()

    logger.info(
        f"[FEATURE] Filtreleme sonrası: {len(df)} etkileşim | "
        f"{df['user_id'].nunique()} kullanıcı | "
        f"{df['post_id'].nunique()} post"
    )

    # Matrisi ve feature'ları oluştur
    matrix, user_idx, item_idx, idx_user, idx_item = build_user_item_matrix(df)
    user_features = build_user_features(df)
    post_features = build_post_features(df)

    return FeatureSet(
        user_item_matrix=matrix,
        user_index=user_idx,
        item_index=item_idx,
        index_user=idx_user,
        index_item=idx_item,
        user_features=user_features,
        post_features=post_features,
        interaction_df=df,
        n_users=matrix.shape[0],
        n_items=matrix.shape[1],
        n_interactions=len(df),
    )
