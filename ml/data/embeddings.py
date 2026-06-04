"""
Embedding Üretici
=================
sentence-transformers ile içerik ve kullanıcı embedding'leri üretir,
PostgreSQL'deki VECTOR(384) kolonlarına yazar.

Desteklenen modeller:
    all-MiniLM-L6-v2     → 384 dim, hızlı, CPU-friendly (varsayılan)
    all-mpnet-base-v2    → 768 dim, daha kaliteli, daha yavaş
    paraphrase-MiniLM-L3 → 384 dim, en hızlı
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# sentence-transformers isteğe bağlı import
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning(
        "sentence-transformers yüklü değil. "
        "Mock embedding kullanılacak. "
        "Kurmak için: pip install sentence-transformers"
    )


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size: int = 64          # GPU'da büyütülebilir
    normalize: bool = True        # L2 normalize — cosine sim için zorunlu
    show_progress: bool = True


class EmbeddingProducer:
    """
    Metin → embedding vektörü dönüştürücü.

    sentence-transformers yüklü değilse deterministik mock
    embedding üretir (test ve demo amaçlı).
    """

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self._model = None

        if ST_AVAILABLE:
            logger.info(f"[EMB] Model yükleniyor: {self.config.model_name}")
            self._model = SentenceTransformer(self.config.model_name)
            logger.info(f"[EMB] Model hazır — dim={self.config.embedding_dim}")
        else:
            logger.info("[EMB] Mock embedding modu aktif")

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Metin listesini embedding matrisine dönüştürür.

        Args:
            texts: encode edilecek metin listesi
        Returns:
            np.ndarray shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)

        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=self.config.show_progress,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
            )
        else:
            embeddings = self._mock_encode(texts)

        logger.info(f"[EMB] {len(texts)} metin encode edildi → shape={embeddings.shape}")
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Tek bir metni encode eder."""
        return self.encode([text])[0]

    def _mock_encode(self, texts: list[str]) -> np.ndarray:
        """
        sentence-transformers olmadan deterministik sahte embedding üretir.
        Hash tabanlı olduğu için aynı metin her zaman aynı vektörü verir.
        """
        dim = self.config.embedding_dim
        result = np.zeros((len(texts), dim), dtype=np.float32)

        for i, text in enumerate(texts):
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(dim).astype(np.float32)
            # L2 normalize
            norm = np.linalg.norm(vec)
            result[i] = vec / norm if norm > 0 else vec

        return result

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """İki vektör arasında cosine benzerliği hesaplar."""
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        dot = np.dot(a, b.T)
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True)
        return float(dot / (norm_a * norm_b.T + 1e-9))

    def batch_cosine_similarity(
        self, query: np.ndarray, candidates: np.ndarray
    ) -> np.ndarray:
        """
        Bir query vektörünü N aday ile karşılaştırır.

        Args:
            query     : shape (dim,)
            candidates: shape (N, dim)
        Returns:
            similarities: shape (N,) — [-1, 1] arası değerler
        """
        q = query / (np.linalg.norm(query) + 1e-9)
        c_norms = np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-9
        c_norm = candidates / c_norms
        return c_norm @ q

    def compute_user_embedding(
        self,
        liked_post_embeddings: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Kullanıcının beğendiği postların embedding'lerinden
        ağırlıklı ortalama ile kullanıcı embedding'i üretir.

        Args:
            liked_post_embeddings: shape (N, dim) — beğenilen post vektörleri
            weights: shape (N,) — her postun ağırlığı (sinyal skoru)
        Returns:
            user_embedding: shape (dim,)
        """
        if len(liked_post_embeddings) == 0:
            return np.zeros(self.config.embedding_dim, dtype=np.float32)

        if weights is None:
            weights = np.ones(len(liked_post_embeddings), dtype=np.float32)

        weights = np.array(weights, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-9)

        user_emb = np.average(liked_post_embeddings, axis=0, weights=weights)

        # L2 normalize
        norm = np.linalg.norm(user_emb)
        return user_emb / norm if norm > 0 else user_emb

    def find_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidate_ids: list[str],
        top_k: int = 20,
        exclude_ids: set[str] = None,
    ) -> list[tuple[str, float]]:
        """
        Query embedding'e en yakın K adayı döner.

        Returns:
            [(id, similarity_score), ...] — azalan sırada
        """
        similarities = self.batch_cosine_similarity(query_embedding, candidate_embeddings)

        results = []
        for idx, (cid, sim) in enumerate(zip(candidate_ids, similarities)):
            if exclude_ids and cid in exclude_ids:
                continue
            results.append((cid, float(sim)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ── Embedding tabanlı öneri ──────────────────────────────────
class EmbeddingRecommender:
    """
    Embedding vektörlerine dayalı içerik öneri motoru.
    Gerçek DB yerine in-memory dict ile çalışır (pipeline içi kullanım).
    """

    def __init__(self, producer: EmbeddingProducer):
        self.producer = producer
        # post_id → embedding vektörü
        self._post_embeddings: dict[str, np.ndarray] = {}
        # post_id → metadata
        self._post_meta: dict[str, dict] = {}

    def index_posts(self, posts: list[dict]):
        """
        Post listesini encode edip in-memory index'e ekler.

        Args:
            posts: [{"id", "content", "media_type", "virality_score", ...}, ...]
        """
        logger.info(f"[EMB-REC] {len(posts)} post indexleniyor")

        texts = [p["content"] for p in posts]
        embeddings = self.producer.encode(texts)

        for post, emb in zip(posts, embeddings):
            self._post_embeddings[post["id"]] = emb
            self._post_meta[post["id"]] = {
                "content": post["content"],
                "media_type": post.get("media_type", "text"),
                "virality_score": post.get("virality_score", 0.0),
                "author": post.get("author", "?"),
            }

        logger.info(f"[EMB-REC] Index boyutu: {len(self._post_embeddings)}")

    def build_user_profile(
        self,
        user_id: str,
        interactions: list[dict],
    ) -> Optional[np.ndarray]:
        """
        Kullanıcının etkileşim geçmişinden ilgi vektörü üretir.

        Args:
            user_id     : hedef kullanıcı
            interactions: [{"post_id", "type", "interaction_score"}, ...]
        Returns:
            user_embedding: shape (dim,) veya None (yeterli veri yoksa)
        """
        positive = [
            i for i in interactions
            if i["user_id"] == user_id
            and i.get("interaction_score", 0) > 0
        ]

        if not positive:
            logger.warning(f"[EMB-REC] {user_id[:8]} için etkileşim bulunamadı")
            return None

        post_embeddings = []
        weights = []

        for inter in positive:
            pid = inter["post_id"]
            if pid in self._post_embeddings:
                post_embeddings.append(self._post_embeddings[pid])
                weights.append(inter.get("interaction_score", 1.0))

        if not post_embeddings:
            return None

        return self.producer.compute_user_embedding(
            np.array(post_embeddings), np.array(weights)
        )

    def recommend(
        self,
        user_id: str,
        interactions: list[dict],
        top_k: int = 20,
        exclude_seen: bool = True,
    ) -> list[dict]:
        """
        Embedding benzerliğine dayalı kişiselleştirilmiş öneri.

        Returns:
            [{"post_id", "author", "content", "media_type",
              "similarity", "virality_score", "final_score"}, ...]
        """
        user_emb = self.build_user_profile(user_id, interactions)
        if user_emb is None:
            return []

        seen_ids = set()
        if exclude_seen:
            seen_ids = {i["post_id"] for i in interactions if i["user_id"] == user_id}

        candidate_ids = list(self._post_embeddings.keys())
        candidate_embs = np.array([self._post_embeddings[pid] for pid in candidate_ids])

        similar = self.producer.find_similar(
            user_emb, candidate_embs, candidate_ids,
            top_k=top_k * 2, exclude_ids=seen_ids
        )

        results = []
        for post_id, sim in similar[:top_k]:
            meta = self._post_meta[post_id]
            virality = min(meta["virality_score"] / 100, 1.0)
            # Embedding sim + virality hybrid
            final_score = sim * 0.75 + virality * 0.25

            results.append({
                "post_id":       post_id[:8],
                "author":        meta["author"],
                "content":       meta["content"][:70],
                "media_type":    meta["media_type"],
                "similarity":    round(float(sim), 4),
                "virality_score": meta["virality_score"],
                "final_score":   round(final_score, 4),
                "source":        "embedding",
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        logger.info(f"[EMB-REC] {len(results)} öneri üretildi — user={user_id[:8]}")
        return results
