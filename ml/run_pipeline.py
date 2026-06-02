"""
ALS — Alternating Least Squares
================================
Implicit feedback (beğeni, görüntüleme, paylaşım) üzerinden
matris faktörizasyonu yapar.

implicit kütüphanesi varsa kullanır; yoksa saf NumPy ile
mini ALS implementasyonu çalışır.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from ..data.feature_engineering import FeatureSet

logger = logging.getLogger(__name__)

try:
    from implicit.als import AlternatingLeastSquares
    from implicit.nearest_neighbours import bm25_weight
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False
    logger.warning("implicit kütüphanesi yok — NumPy ALS kullanılacak.")


@dataclass
class ALSConfig:
    factors: int = 64            # latent factor boyutu
    iterations: int = 20         # ALS iterasyon sayısı
    regularization: float = 0.01 # L2 regularizasyon
    alpha: float = 40.0          # confidence ölçekleme (BM25 yoksa)
    use_bm25: bool = True        # BM25 ağırlıklandırma
    random_state: int = 42
    # implicit kütüphanesi varsa GPU kullanılsın mı?
    use_gpu: bool = False


@dataclass
class ALSResult:
    user_factors: np.ndarray     # shape (n_users, factors)
    item_factors: np.ndarray     # shape (n_items, factors)
    train_time_s: float
    config: ALSConfig
    model_version: str = "als_v1"

    def predict_score(self, user_idx: int, item_idx: int) -> float:
        """Tek bir (user, item) çifti için skor tahmin eder."""
        return float(self.user_factors[user_idx] @ self.item_factors[item_idx])

    def recommend_for_user(
        self,
        user_idx: int,
        n: int = 20,
        exclude_indices: set[int] = None,
    ) -> list[tuple[int, float]]:
        """
        Bir kullanıcı için en iyi N item önerir.

        Returns:
            [(item_idx, score), ...] — azalan sırada
        """
        scores = self.item_factors @ self.user_factors[user_idx]

        if exclude_indices:
            scores[list(exclude_indices)] = -np.inf

        top_indices = np.argpartition(scores, -n)[-n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(int(i), float(scores[i])) for i in top_indices]


# ── NumPy tabanlı mini ALS (implicit yoksa) ────────────────────
class _NumpyALS:
    """
    Basit ALS implementasyonu.
    Büyük matrisler için implicit kullan; bu sınıf demo/test içindir.
    """

    def __init__(self, config: ALSConfig):
        self.cfg = config
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def fit(self, matrix: csr_matrix) -> None:
        rng = np.random.RandomState(self.cfg.random_state)
        n_users, n_items = matrix.shape
        f = self.cfg.factors
        alpha = self.cfg.alpha
        lam = self.cfg.regularization

        # Başlangıç faktörleri
        U = rng.randn(n_users, f).astype(np.float32) * 0.01
        V = rng.randn(n_items, f).astype(np.float32) * 0.01

        # Confidence matrisi: C = 1 + alpha * R
        R = matrix.toarray().astype(np.float32)
        C = 1.0 + alpha * R

        I_f = np.eye(f, dtype=np.float32) * lam

        for it in range(self.cfg.iterations):
            # Kullanıcı faktörlerini güncelle
            VtV = V.T @ V
            for u in range(n_users):
                c_u = np.diag(C[u])
                A = VtV + V.T @ (c_u - np.eye(n_items)) @ V + I_f
                b = V.T @ c_u @ R[u]
                U[u] = np.linalg.solve(A, b)

            # Item faktörlerini güncelle
            UtU = U.T @ U
            for i in range(n_items):
                c_i = np.diag(C[:, i])
                A = UtU + U.T @ (c_i - np.eye(n_users)) @ U + I_f
                b = U.T @ c_i @ R[:, i]
                V[i] = np.linalg.solve(A, b)

            if (it + 1) % 5 == 0:
                # Basit loss hesabı
                pred = U @ V.T
                loss = np.sum(C * (R - pred) ** 2) + lam * (np.sum(U**2) + np.sum(V**2))
                logger.debug(f"  ALS iter {it+1}/{self.cfg.iterations} — loss={loss:.2f}")

        self.user_factors = U
        self.item_factors = V


# ── Ana ALS Trainer ────────────────────────────────────────────
class ALSTrainer:
    """
    ALS modeli eğitimi. implicit kütüphanesini otomatik kullanır;
    yoksa NumPy fallback devreye girer.
    """

    def __init__(self, config: ALSConfig = None):
        self.config = config or ALSConfig()

    def train(self, feature_set: FeatureSet) -> ALSResult:
        """
        FeatureSet'ten ALS modeli eğitir.

        Args:
            feature_set: prepare_features() çıktısı
        Returns:
            ALSResult: eğitilmiş faktör matrisleri
        """
        matrix = feature_set.user_item_matrix
        logger.info(
            f"[ALS] Eğitim başlıyor — "
            f"{matrix.shape[0]} kullanıcı × {matrix.shape[1]} item | "
            f"factors={self.config.factors} | iters={self.config.iterations}"
        )

        t0 = time.time()

        if IMPLICIT_AVAILABLE:
            user_factors, item_factors = self._train_implicit(matrix)
        else:
            user_factors, item_factors = self._train_numpy(matrix)

        elapsed = time.time() - t0
        logger.info(f"[ALS] Eğitim tamamlandı — {elapsed:.1f}s")

        return ALSResult(
            user_factors=user_factors,
            item_factors=item_factors,
            train_time_s=elapsed,
            config=self.config,
        )

    def _train_implicit(self, matrix: csr_matrix):
        if self.config.use_bm25:
            matrix = bm25_weight(matrix, K1=100, B=0.8).tocsr()
            logger.info("[ALS] BM25 ağırlıklandırma uygulandı")
        else:
            matrix = (matrix * self.config.alpha).tocsr()

        model = AlternatingLeastSquares(
            factors=self.config.factors,
            iterations=self.config.iterations,
            regularization=self.config.regularization,
            use_gpu=self.config.use_gpu,
            random_state=self.config.random_state,
        )
        model.fit(matrix)
        return model.user_factors, model.item_factors

    def _train_numpy(self, matrix: csr_matrix):
        # Büyük matrisler için NumPy ALS yavaştır; demo amaçlıdır
        als = _NumpyALS(self.config)
        als.fit(matrix)
        return als.user_factors, als.item_factors

    def recommend(
        self,
        result: ALSResult,
        feature_set: FeatureSet,
        user_id: str,
        top_k: int = 20,
    ) -> list[dict]:
        """
        Eğitilmiş ALS modeliyle kullanıcıya öneri üretir.

        Returns:
            [{"post_id", "score", "rank", "source"}, ...]
        """
        if user_id not in feature_set.user_index:
            logger.warning(f"[ALS] Kullanıcı bulunamadı: {user_id[:8]} — cold start")
            return []

        u_idx = feature_set.user_index[user_id]

        # Kullanıcının zaten gördüğü itemleri hariç tut
        seen_indices = set(feature_set.user_item_matrix[u_idx].nonzero()[1].tolist())

        recs = result.recommend_for_user(u_idx, n=top_k + len(seen_indices), exclude_indices=seen_indices)

        results = []
        for rank, (item_idx, score) in enumerate(recs[:top_k], 1):
            post_id = feature_set.index_item.get(item_idx, "")
            if not post_id:
                continue
            results.append({
                "post_id": post_id,
                "score":   round(float(score), 4),
                "rank":    rank,
                "source":  "als",
            })

        logger.info(f"[ALS] {len(results)} öneri — user={user_id[:8]}")
        return results
