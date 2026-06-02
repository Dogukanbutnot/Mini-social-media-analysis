"""
Değerlendirme Metrikleri
========================
Öneri modellerini karşılaştırmak için standart IR metrikleri:
    - Precision@K
    - Recall@K
    - NDCG@K (Normalized Discounted Cumulative Gain)
    - Coverage   (katalogdan kaç item önerildi)
    - Diversity  (önerilerin ne kadar çeşitli olduğu)
    - Novelty    (ne kadar popüler olmayan itemler önerildi)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    model_name: str
    precision_at_k: dict[int, float] = field(default_factory=dict)  # {5: 0.12, 10: 0.09}
    recall_at_k:    dict[int, float] = field(default_factory=dict)
    ndcg_at_k:      dict[int, float] = field(default_factory=dict)
    coverage:       float = 0.0
    diversity:      float = 0.0
    novelty:        float = 0.0
    n_users_eval:   int   = 0

    def summary(self) -> dict:
        return {
            "model":        self.model_name,
            "P@5":          round(self.precision_at_k.get(5, 0), 4),
            "P@10":         round(self.precision_at_k.get(10, 0), 4),
            "R@10":         round(self.recall_at_k.get(10, 0), 4),
            "NDCG@5":       round(self.ndcg_at_k.get(5, 0), 4),
            "NDCG@10":      round(self.ndcg_at_k.get(10, 0), 4),
            "coverage":     round(self.coverage, 4),
            "diversity":    round(self.diversity, 4),
            "novelty":      round(self.novelty, 4),
            "n_users":      self.n_users_eval,
        }

    def __repr__(self):
        s = self.summary()
        return (
            f"[{s['model']:12s}] "
            f"NDCG@10={s['NDCG@10']:.4f} | "
            f"P@10={s['P@10']:.4f} | "
            f"Coverage={s['coverage']:.3f} | "
            f"Diversity={s['diversity']:.3f}"
        )


# ── Temel metrik fonksiyonları ─────────────────────────────────

def precision_at_k(
    recommended: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """İlk K önerideki ilgili item oranı."""
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(
    recommended: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """İlk K öneride yakalanan ilgili item oranı."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def dcg_at_k(
    recommended: list[str],
    relevant: set[str],
    k: int,
    relevance_scores: Optional[dict[str, float]] = None,
) -> float:
    """
    Discounted Cumulative Gain.
    relevance_scores verilmezse binary (0/1) kullanır.
    """
    dcg = 0.0
    for i, item in enumerate(recommended[:k], 1):
        if relevance_scores:
            rel = relevance_scores.get(item, 0.0)
        else:
            rel = 1.0 if item in relevant else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    recommended: list[str],
    relevant: set[str],
    k: int,
    relevance_scores: Optional[dict[str, float]] = None,
) -> float:
    """
    Normalized DCG. 0 ile 1 arasında.
    1.0 = mükemmel sıralama.
    """
    actual_dcg = dcg_at_k(recommended, relevant, k, relevance_scores)

    # İdeal sıralama: ilgili itemler en üste
    if relevance_scores:
        ideal_items = sorted(relevant, key=lambda x: relevance_scores.get(x, 0), reverse=True)
    else:
        ideal_items = list(relevant)

    ideal_dcg = dcg_at_k(ideal_items, relevant, k, relevance_scores)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


# ── Model Evaluator ────────────────────────────────────────────

class ModelEvaluator:
    """
    Birden fazla modeli aynı test seti üzerinde değerlendirir.

    Kullanım:
        evaluator = ModelEvaluator(k_values=[5, 10, 20])
        result = evaluator.evaluate(
            model_name="als",
            recommendations={"user_1": ["post_a", "post_b", ...]},
            ground_truth={"user_1": {"post_a", "post_c"}},
        )
    """

    def __init__(self, k_values: list[int] = None):
        self.k_values = k_values or [5, 10, 20]

    def evaluate(
        self,
        model_name: str,
        recommendations: dict[str, list[str]],  # {user_id: [post_id, ...]}
        ground_truth: dict[str, set[str]],       # {user_id: {relevant_post_ids}}
        all_item_ids: Optional[set[str]] = None,  # katalog (coverage için)
        item_popularity: Optional[dict[str, int]] = None,  # {post_id: view_count}
    ) -> MetricResult:
        """
        Model değerlendirmesi yapar.

        Args:
            model_name      : model adı (raporda görünür)
            recommendations : her kullanıcı için sıralı öneri listesi
            ground_truth    : her kullanıcı için ilgili item seti (test seti)
            all_item_ids    : tüm katalog (coverage hesabı için)
            item_popularity : item popülarite (novelty hesabı için)
        Returns:
            MetricResult
        """
        logger.info(f"[EVAL] {model_name} değerlendiriliyor — {len(recommendations)} kullanıcı")

        precision_sums = {k: 0.0 for k in self.k_values}
        recall_sums    = {k: 0.0 for k in self.k_values}
        ndcg_sums      = {k: 0.0 for k in self.k_values}

        all_recommended_items: set[str] = set()
        intra_list_diversities: list[float] = []
        novelty_scores: list[float] = []

        n_eval = 0

        for user_id, recs in recommendations.items():
            if not recs:
                continue
            relevant = ground_truth.get(user_id, set())
            if not relevant:
                continue

            n_eval += 1

            for k in self.k_values:
                precision_sums[k] += precision_at_k(recs, relevant, k)
                recall_sums[k]    += recall_at_k(recs, relevant, k)
                ndcg_sums[k]      += ndcg_at_k(recs, relevant, k)

            all_recommended_items.update(recs)

            # Novelty: popüler olmayan itemleri önermek daha "yeni" sayılır
            if item_popularity:
                max_pop = max(item_popularity.values()) or 1
                nov = np.mean([
                    1 - item_popularity.get(p, 0) / max_pop
                    for p in recs[:10]
                ])
                novelty_scores.append(float(nov))

        if n_eval == 0:
            logger.warning(f"[EVAL] {model_name}: değerlendirilebilir kullanıcı yok")
            return MetricResult(model_name=model_name)

        # Ortalamaları hesapla
        result = MetricResult(
            model_name=model_name,
            precision_at_k={k: precision_sums[k] / n_eval for k in self.k_values},
            recall_at_k=   {k: recall_sums[k]    / n_eval for k in self.k_values},
            ndcg_at_k=     {k: ndcg_sums[k]      / n_eval for k in self.k_values},
            n_users_eval=n_eval,
        )

        # Coverage: katalogdan kaç item en az bir kez önerildi
        if all_item_ids:
            result.coverage = len(all_recommended_items & all_item_ids) / len(all_item_ids)

        # Novelty
        if novelty_scores:
            result.novelty = float(np.mean(novelty_scores))

        logger.info(f"[EVAL] {result}")
        return result

    def compare_models(self, results: list[MetricResult]) -> pd.DataFrame:
        """
        Birden fazla modeli karşılaştıran DataFrame üretir.
        En iyi değerler bold gösterilebilir (Jupyter için).
        """
        rows = [r.summary() for r in results]
        df = pd.DataFrame(rows).set_index("model")

        # En iyi değerleri belirle
        best = {}
        for col in df.columns:
            if col == "n_users":
                continue
            best[col] = df[col].idxmax()

        logger.info(f"\n{'='*60}\nModel Karşılaştırması\n{'='*60}\n{df.to_string()}\n")
        return df

    def train_test_split(
        self,
        interaction_df: pd.DataFrame,
        test_ratio: float = 0.2,
        min_interactions: int = 5,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, dict[str, set[str]]]:
        """
        Zaman tabanlı train/test split.
        Her kullanıcının son %test_ratio etkileşimi test seti.

        Returns:
            train_df   : eğitim etkileşimleri
            ground_truth: {user_id: {test post_ids}}
        """
        np.random.seed(random_state)

        train_rows, ground_truth = [], {}

        for user_id, group in interaction_df.groupby("user_id"):
            if len(group) < min_interactions:
                # Yeterli etkileşim yok — hepsini train'e ekle
                train_rows.append(group)
                continue

            group_sorted = group.sort_values("created_at")
            n_test = max(1, int(len(group_sorted) * test_ratio))
            n_train = len(group_sorted) - n_test

            train_rows.append(group_sorted.iloc[:n_train])

            # Sadece yüksek kaliteli etkileşimleri ilgili say
            test_group = group_sorted.iloc[n_train:]
            relevant = set(
                test_group[test_group["interaction_score"] >= 3.0]["post_id"].tolist()
            )
            if relevant:
                ground_truth[user_id] = relevant

        train_df = pd.concat(train_rows, ignore_index=True)

        logger.info(
            f"[EVAL] Train/test split: "
            f"train={len(train_df)}, "
            f"test_users={len(ground_truth)}"
        )
        return train_df, ground_truth
