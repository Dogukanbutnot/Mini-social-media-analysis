"""
SVD — Truncated Singular Value Decomposition
=============================================
sklearn'ın TruncatedSVD'si ile matris faktörizasyonu.
Explicit/implicit feedback her ikisinde de çalışır.
ALS'e göre daha hızlı; doğruluğu implicit'te biraz düşük olabilir.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from ..data.feature_engineering import FeatureSet

logger = logging.getLogger(__name__)


@dataclass
class SVDConfig:
    n_components: int = 64       # latent faktör sayısı
    n_iter: int = 10             # randomized SVD iterasyonu
    random_state: int = 42
    normalize_embeddings: bool = True  # L2 normalize
    model_version: str = "svd_v1"


@dataclass
class SVDResult:
    user_embeddings: np.ndarray  # shape (n_users, n_components)
    item_embeddings: np.ndarray  # shape (n_items, n_components)
    singular_values: np.ndarray  # shape (n_components,)
    explained_variance_ratio: np.ndarray
    train_time_s: float
    config: SVDConfig

    @property
    def model_version(self) -> str:
        return self.config.model_version

    def predict_score(self, user_idx: int, item_idx: int) -> float:
        return float(self.user_embeddings[user_idx] @ self.item_embeddings[item_idx])

    def recommend_for_user(
        self,
        user_idx: int,
        n: int = 20,
        exclude_indices: set[int] = None,
    ) -> list[tuple[int, float]]:
        """Kullanıcı embedding'iyle item embedding'lerinin dot product'ı."""
        scores = self.item_embeddings @ self.user_embeddings[user_idx]

        if exclude_indices:
            scores[list(exclude_indices)] = -np.inf

        top_indices = np.argpartition(scores, -n)[-n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(int(i), float(scores[i])) for i in top_indices]

    def get_similar_items(
        self,
        item_idx: int,
        n: int = 10,
        exclude_self: bool = True,
    ) -> list[tuple[int, float]]:
        """Item-item benzerliği (cosine similarity üzerinden)."""
        query = self.item_embeddings[item_idx]
        sims = self.item_embeddings @ query

        if exclude_self:
            sims[item_idx] = -np.inf

        top_indices = np.argpartition(sims, -n)[-n:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
        return [(int(i), float(sims[i])) for i in top_indices]


class SVDTrainer:
    """
    TruncatedSVD ile matris faktörizasyonu.

    Kullanım:
        trainer = SVDTrainer(SVDConfig(n_components=128))
        result  = trainer.train(feature_set)
        recs    = trainer.recommend(result, feature_set, user_id)
    """

    def __init__(self, config: SVDConfig = None):
        self.config = config or SVDConfig()

    def train(self, feature_set: FeatureSet) -> SVDResult:
        """
        FeatureSet'ten SVD modeli eğitir.

        Returns:
            SVDResult: kullanıcı ve item embedding matrisleri
        """
        matrix = feature_set.user_item_matrix
        logger.info(
            f"[SVD] Eğitim başlıyor — "
            f"{matrix.shape} | components={self.config.n_components}"
        )

        t0 = time.time()

        svd = TruncatedSVD(
            n_components=self.config.n_components,
            n_iter=self.config.n_iter,
            random_state=self.config.random_state,
        )

        # Item embedding: V^T * Σ
        # Kullanıcı embedding: U * Σ
        user_emb = svd.fit_transform(matrix)            # (n_users, k)
        item_emb = svd.components_.T                    # (n_items, k)

        # Sigma'yı kullanıcı tarafına absorbe et
        sigma = svd.singular_values_
        item_emb = item_emb * sigma[np.newaxis, :]      # (n_items, k)

        if self.config.normalize_embeddings:
            user_emb = normalize(user_emb, norm="l2")
            item_emb = normalize(item_emb, norm="l2")

        elapsed = time.time() - t0

        ev_ratio = svd.explained_variance_ratio_
        logger.info(
            f"[SVD] Tamamlandı — {elapsed:.1f}s | "
            f"açıklanan varyans: {ev_ratio.sum():.3f} "
            f"(top-10: {ev_ratio[:10].sum():.3f})"
        )

        return SVDResult(
            user_embeddings=user_emb.astype(np.float32),
            item_embeddings=item_emb.astype(np.float32),
            singular_values=sigma.astype(np.float32),
            explained_variance_ratio=ev_ratio,
            train_time_s=elapsed,
            config=self.config,
        )

    def recommend(
        self,
        result: SVDResult,
        feature_set: FeatureSet,
        user_id: str,
        top_k: int = 20,
    ) -> list[dict]:
        """
        Eğitilmiş SVD modeliyle öneri üretir.

        Returns:
            [{"post_id", "score", "rank", "source"}, ...]
        """
        if user_id not in feature_set.user_index:
            logger.warning(f"[SVD] Kullanıcı bulunamadı: {user_id[:8]}")
            return []

        u_idx = feature_set.user_index[user_id]
        seen_indices = set(feature_set.user_item_matrix[u_idx].nonzero()[1].tolist())

        recs = result.recommend_for_user(
            u_idx, n=top_k + len(seen_indices), exclude_indices=seen_indices
        )

        results = []
        for rank, (item_idx, score) in enumerate(recs[:top_k], 1):
            post_id = feature_set.index_item.get(item_idx, "")
            if not post_id:
                continue
            results.append({
                "post_id": post_id,
                "score":   round(float(score), 4),
                "rank":    rank,
                "source":  "svd",
            })

        logger.info(f"[SVD] {len(results)} öneri — user={user_id[:8]}")
        return results

    def get_similar_posts(
        self,
        result: SVDResult,
        feature_set: FeatureSet,
        post_id: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Item-item benzerliği: verilen posta en benzer postları döner.
        "Bunu beğenenler bunları da beğendi" senaryosu.
        """
        if post_id not in feature_set.item_index:
            return []

        i_idx = feature_set.item_index[post_id]
        similar = result.get_similar_items(i_idx, n=top_k)

        return [
            {
                "post_id":    feature_set.index_item.get(item_idx, ""),
                "similarity": round(float(sim), 4),
                "rank":       rank,
            }
            for rank, (item_idx, sim) in enumerate(similar, 1)
            if feature_set.index_item.get(item_idx)
        ]
