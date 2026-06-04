"""
ML Pipeline Orkestrasyonu
=========================
Tüm adımları uçtan uca çalıştırır:

  1. Mock veri üret (veya DB'den çek)
  2. Feature engineering
  3. Embedding üretimi (sentence-transformers)
  4. Train/test split
  5. ALS, SVD, LightGBM modellerini eğit
  6. Tüm modelleri değerlendir (NDCG, Precision@K, Coverage)
  7. Karşılaştırma tablosu yaz
  8. Kazanan modeli belirle
"""

import logging
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..data.feature_engineering import prepare_features
from ..data.embeddings import EmbeddingProducer, EmbeddingRecommender, EmbeddingConfig
from ..models.als_model import ALSTrainer, ALSConfig
from ..models.svd_model import SVDTrainer, SVDConfig
from ..models.lgbm_model import LGBMTrainer, LGBMConfig, build_training_dataframe
from ..evaluation.metrics import ModelEvaluator, MetricResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ml_pipeline")


# ════════════════════════════════════════════════════════════
# MOCK VERİ ÜRETİCİ
# ════════════════════════════════════════════════════════════

def generate_mock_interactions(
    n_users: int = 80,
    n_posts: int = 200,
    n_interactions: int = 2000,
    random_seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    ML pipeline için gerçekçi mock veri üretir.

    Returns:
        users        : [{"id", "username"}, ...]
        posts        : [{"id", "content", "media_type", "virality_score"}, ...]
        interactions : [{"user_id", "post_id", "type", "dwell_time_ms",
                          "scroll_depth", "created_at"}, ...]
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Kullanıcılar
    users = [
        {"id": str(uuid.uuid4()), "username": f"user_{i:03d}"}
        for i in range(n_users)
    ]

    # Postlar
    topics = [
        "Python ile makine öğrenmesi — temel kavramlar",
        "Transformer mimarisini anlamak",
        "SQL window functions ile analitik sorgular",
        "Vector database nedir, nasıl kullanılır?",
        "LLM fine-tuning en iyi pratikleri",
        "RAG pipeline kurulum rehberi",
        "Recommendation system tasarımı — adım adım",
        "FastAPI ile production API geliştirme",
        "A/B test istatistiksel anlamlılık hesabı",
        "Graph neural network'lere giriş",
        "Docker ve Kubernetes ile ML deployment",
        "Feature engineering ipuçları",
        "Embedding modelleri nasıl seçilir?",
        "PostgreSQL performans optimizasyonu",
        "PyTorch vs TensorFlow — karşılaştırma",
        "Data pipeline ile ETL otomasyonu",
        "Explainable AI — model yorumlanabilirliği",
        "Reinforcement learning gerçek dünya uygulamaları",
        "Time series forecasting yöntemleri",
        "NLP pipeline tasarımı",
    ]
    media_types = ["text", "image", "video", "reel"]

    posts = []
    for i in range(n_posts):
        topic = topics[i % len(topics)]
        posts.append({
            "id":             str(uuid.uuid4()),
            "content":        f"{topic} — bölüm {i // len(topics) + 1}",
            "media_type":     random.choice(media_types),
            "virality_score": round(random.expovariate(1 / 15), 2),  # üstel dağılım
            "author":         f"author_{random.randint(0, 20)}",
        })

    # Etkileşimler — power-law dağılımlı (gerçekçi)
    user_ids  = [u["id"] for u in users]
    post_ids  = [p["id"] for p in posts]
    int_types = ["like", "view", "save", "share", "comment"]
    weights   = [0.25, 0.45, 0.12, 0.10, 0.08]

    now = datetime.utcnow()
    interactions = []

    for _ in range(n_interactions):
        uid  = random.choice(user_ids)
        pid  = random.choice(post_ids)
        itype = random.choices(int_types, weights=weights, k=1)[0]

        interactions.append({
            "user_id":      uid,
            "post_id":      pid,
            "type":         itype,
            "dwell_time_ms": random.randint(2_000, 90_000) if itype == "view" else 0,
            "scroll_depth": round(random.uniform(0.1, 1.0), 2),
            "created_at":   now - timedelta(hours=random.randint(0, 720)),
        })

    logger.info(
        f"[MOCK] {len(users)} kullanıcı | "
        f"{len(posts)} post | "
        f"{len(interactions)} etkileşim üretildi"
    )
    return users, posts, interactions


# ════════════════════════════════════════════════════════════
# ML PIPELINE
# ════════════════════════════════════════════════════════════

class MLPipeline:
    """
    Uçtan uca ML pipeline.

    Adımlar:
        1. Veri hazırlığı (feature engineering)
        2. Embedding üretimi
        3. Train/test split
        4. Model eğitimi (ALS + SVD + LightGBM)
        5. Değerlendirme (NDCG, P@K, Coverage)
        6. Karşılaştırma raporu
    """

    def __init__(self):
        self.evaluator = ModelEvaluator(k_values=[5, 10, 20])
        self.emb_producer = EmbeddingProducer(EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            show_progress=False,
        ))
        self.als_trainer  = ALSTrainer(ALSConfig(factors=32, iterations=15))
        self.svd_trainer  = SVDTrainer(SVDConfig(n_components=32))
        self.lgbm_trainer = LGBMTrainer(LGBMConfig(n_estimators=100, verbose=-1))

    def run(
        self,
        users: list[dict],
        posts: list[dict],
        interactions: list[dict],
    ) -> dict:
        """
        Pipeline'ı çalıştırır ve karşılaştırma raporunu döner.
        """
        print("\n" + "═" * 65)
        print("  ML PIPELINE BAŞLIYOR")
        print("═" * 65)

        # ── 1. Feature Engineering ────────────────────────
        print("\n📐 ADIM 1: Feature Engineering")
        feature_set = prepare_features(interactions)
        print(
            f"   ✓ {feature_set.n_users} kullanıcı × "
            f"{feature_set.n_items} item matrisi | "
            f"density={feature_set.user_item_matrix.nnz / (feature_set.n_users * feature_set.n_items + 1e-9):.3%}"
        )

        # ── 2. Embedding Üretimi ──────────────────────────
        print("\n🔢 ADIM 2: Embedding Üretimi (sentence-transformers)")
        emb_recommender = EmbeddingRecommender(self.emb_producer)
        emb_recommender.index_posts(posts)

        # Kullanıcı ilgi vektörleri
        # interaction_df'i dict listesine çevir (embedding profil için)
        interaction_dicts = feature_set.interaction_df.to_dict("records")
        user_embeddings = {}
        for u in users:
            emb = emb_recommender.build_user_profile(u["id"], interaction_dicts)
            if emb is not None:
                user_embeddings[u["id"]] = emb
        print(f"   ✓ {len(posts)} post indexlendi | {len(user_embeddings)} kullanıcı profili üretildi")

        # ── 3. Train/Test Split ───────────────────────────
        print("\n✂️  ADIM 3: Train/Test Split (zaman tabanlı, %80/%20)")
        train_df, ground_truth = self.evaluator.train_test_split(
            feature_set.interaction_df, test_ratio=0.2
        )
        print(f"   ✓ Train: {len(train_df)} | Test kullanıcı: {len(ground_truth)}")

        # Train feature set (sadece train verisi ile)
        train_feature_set = prepare_features(train_df.to_dict("records"))

        all_post_ids = {p["id"] for p in posts}
        # post_features index=post_id
        item_popularity = feature_set.post_features["total_interactions"].to_dict()

        results: list[MetricResult] = []

        # ── 4a. ALS ──────────────────────────────────────
        print("\n🔄 ADIM 4a: ALS Eğitimi")
        als_result = self.als_trainer.train(train_feature_set)
        als_recs: dict[str, list[str]] = {}
        for uid in list(ground_truth.keys())[:50]:   # ilk 50 kullanıcı
            recs = self.als_trainer.recommend(als_result, train_feature_set, uid, top_k=20)
            als_recs[uid] = [r["post_id"] for r in recs]

        als_metric = self.evaluator.evaluate(
            "ALS", als_recs, ground_truth, all_post_ids, item_popularity
        )
        results.append(als_metric)
        print(f"   ✓ {als_metric}")

        # ── 4b. SVD ──────────────────────────────────────
        print("\n🔵 ADIM 4b: SVD Eğitimi")
        svd_result = self.svd_trainer.train(train_feature_set)
        svd_recs: dict[str, list[str]] = {}
        for uid in list(ground_truth.keys())[:50]:
            recs = self.svd_trainer.recommend(svd_result, train_feature_set, uid, top_k=20)
            svd_recs[uid] = [r["post_id"] for r in recs]

        svd_metric = self.evaluator.evaluate(
            "SVD", svd_recs, ground_truth, all_post_ids, item_popularity
        )
        results.append(svd_metric)
        print(f"   ✓ {svd_metric}")

        # ── 4c. LightGBM ─────────────────────────────────
        print("\n🌿 ADIM 4c: LightGBM Eğitimi")
        train_feature_df = build_training_dataframe(train_feature_set)
        # Val split
        val_mask = np.random.rand(len(train_feature_df)) < 0.15
        lgbm_result = self.lgbm_trainer.train(
            train_feature_df[~val_mask],
            train_feature_df[val_mask] if val_mask.sum() > 0 else None,
        )

        lgbm_recs: dict[str, list[str]] = {}
        for uid in list(ground_truth.keys())[:50]:
            recs = self.lgbm_trainer.recommend(lgbm_result, train_feature_set, uid, top_k=20)
            lgbm_recs[uid] = [r["post_id"] for r in recs]

        lgbm_metric = self.evaluator.evaluate(
            "LightGBM", lgbm_recs, ground_truth, all_post_ids, item_popularity
        )
        results.append(lgbm_metric)
        print(f"   ✓ {lgbm_metric}")

        # ── 4d. Embedding (CBF) ───────────────────────────
        print("\n🧠 ADIM 4d: Embedding Recommender Değerlendirmesi")
        emb_recs: dict[str, list[str]] = {}
        emb_interaction_dicts = feature_set.interaction_df.to_dict("records")
        for uid in list(ground_truth.keys())[:50]:
            recs = emb_recommender.recommend(uid, emb_interaction_dicts, top_k=20)
            emb_recs[uid] = [r["post_id"] for r in recs]

        emb_metric = self.evaluator.evaluate(
            "Embedding-CBF", emb_recs, ground_truth, all_post_ids, item_popularity
        )
        results.append(emb_metric)
        print(f"   ✓ {emb_metric}")

        # ── 5. Karşılaştırma Raporu ───────────────────────
        print("\n📊 ADIM 5: Model Karşılaştırma Raporu")
        comparison_df = self.evaluator.compare_models(results)
        print("\n" + comparison_df.to_string())

        # ── 6. Kazanan ────────────────────────────────────
        best_model = max(results, key=lambda r: r.ndcg_at_k.get(10, 0))
        print(f"\n🏆 KAZANAN: {best_model.model_name} "
              f"(NDCG@10={best_model.ndcg_at_k.get(10, 0):.4f})")

        # ── 7. Feature Importance (LightGBM) ─────────────
        print("\n📈 LightGBM Feature Importance (Top 10):")
        top_features = sorted(
            lgbm_result.feature_importances.items(),
            key=lambda x: x[1], reverse=True
        )[:10]
        for i, (feat, imp) in enumerate(top_features, 1):
            bar = "█" * max(1, int(imp / max(v for _, v in top_features) * 20))
            print(f"   {i:2d}. {feat:35s} {bar} {imp:.1f}")

        print("\n" + "═" * 65)
        print("  PIPELINE TAMAMLANDI")
        print("═" * 65 + "\n")

        return {
            "feature_set":     feature_set,
            "als_result":      als_result,
            "svd_result":      svd_result,
            "lgbm_result":     lgbm_result,
            "emb_recommender": emb_recommender,
            "comparison_df":   comparison_df,
            "best_model":      best_model.model_name,
            "metrics":         [r.summary() for r in results],
        }


# ════════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════════

def main():
    users, posts, interactions = generate_mock_interactions(
        n_users=80, n_posts=200, n_interactions=2500
    )
    pipeline = MLPipeline()
    report = pipeline.run(users, posts, interactions)

    print(f"En iyi model: {report['best_model']}")
    print("\nMetrik özeti:")
    metrics_df = pd.DataFrame(report["metrics"]).set_index("model")
    print(metrics_df[["NDCG@5", "NDCG@10", "P@10", "coverage", "novelty"]].to_string())


if __name__ == "__main__":
    main()
