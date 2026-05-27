"""
LightGBM — Feature-based Learning to Rank
==========================================
ALS/SVD'nin ürettiği latent skorları + el ile tasarlanmış
feature'ları birleştirerek LightGBM ile ranking modeli eğitir.

Eğitim sinyali: was_clicked (binary relevance)
Görev: LambdaRank / pointwise classification
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..data.feature_engineering import FeatureSet

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("lightgbm yüklü değil — mock model kullanılacak.")


@dataclass
class LGBMConfig:
    # Model hiperparametreleri
    objective: str = "binary"           # binary: CTR tahmini
    metric: str = "auc"
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1             # L1
    reg_lambda: float = 0.1            # L2
    random_state: int = 42
    early_stopping_rounds: int = 20
    verbose: int = -1
    # Feature mühendisliği
    use_als_scores: bool = True
    use_svd_scores: bool = True
    use_content_features: bool = True
    model_version: str = "lgbm_v1"


@dataclass
class LGBMResult:
    model: object                       # lgb.Booster veya mock
    feature_names: list[str]
    feature_importances: dict[str, float]
    train_auc: float
    val_auc: Optional[float]
    train_time_s: float
    config: LGBMConfig
    n_train_samples: int

    @property
    def model_version(self) -> str:
        return self.config.model_version

    def __repr__(self):
        top5 = sorted(
            self.feature_importances.items(), key=lambda x: x[1], reverse=True
        )[:5]
        return (
            f"LGBMResult(train_auc={self.train_auc:.4f}, "
            f"val_auc={self.val_auc}, "
            f"top_features={[f for f, _ in top5]})"
        )


def build_training_dataframe(
    feature_set: FeatureSet,
    als_scores: Optional[dict] = None,   # {(user_id, post_id): score}
    svd_scores: Optional[dict] = None,
    clicked_pairs: Optional[set] = None, # {(user_id, post_id)} — pozitif etiketler
) -> pd.DataFrame:
    """
    LightGBM eğitim DataFrame'i oluşturur.

    Her satır bir (user, item) çiftidir; label = tıklama (0/1).

    Args:
        feature_set  : prepare_features() çıktısı
        als_scores   : ALS model skorları {(uid, pid): score}
        svd_scores   : SVD model skorları {(uid, pid): score}
        clicked_pairs: Gerçekte tıklanan (uid, pid) çiftleri
    Returns:
        DataFrame: feature kolonları + label kolonu
    """
    logger.info("[LGBM] Eğitim DataFrame'i oluşturuluyor")

    interaction_df = feature_set.interaction_df
    user_feats = feature_set.user_features
    post_feats = feature_set.post_features

    records = []

    # Her etkileşimi bir satır olarak al
    for _, row in interaction_df.iterrows():
        uid = row["user_id"]
        pid = row["post_id"]

        record = {
            "user_id": uid,
            "post_id": pid,
            # Label: gerçek tıklama / yüksek engagement
            "label": 1 if (
                clicked_pairs and (uid, pid) in clicked_pairs
                or row["interaction_score"] >= 3.0   # like veya üzeri
            ) else 0,
        }

        # Kullanıcı özellikleri
        if uid in user_feats.index:
            u = user_feats.loc[uid]
            record.update({
                "u_total_interactions": u.get("total_interactions", 0),
                "u_like_rate":          u.get("like_rate", 0),
                "u_share_rate":         u.get("share_rate", 0),
                "u_save_rate":          u.get("save_rate", 0),
                "u_avg_dwell_ms":       u.get("avg_dwell_ms", 0),
                "u_avg_scroll":         u.get("avg_scroll", 0),
                "u_recency_days":       u.get("recency_days", 999),
                "u_log_interactions":   u.get("log_interactions", 0),
            })

        # Post özellikleri
        if pid in post_feats.index:
            p = post_feats.loc[pid]
            record.update({
                "p_unique_users":     p.get("unique_users", 0),
                "p_popularity_score": p.get("popularity_score", 0),
                "p_like_rate":        p.get("like_rate", 0),
                "p_share_rate":       p.get("share_rate", 0),
                "p_avg_dwell_ms":     p.get("avg_dwell_ms", 0),
                "p_freshness":        p.get("freshness", 0),
                "p_post_age_days":    p.get("post_age_days", 0),
                "p_log_unique_users": p.get("log_unique_users", 0),
            })

        # Model skorları (stacking feature olarak)
        if als_scores:
            record["als_score"] = als_scores.get((uid, pid), 0.0)
        if svd_scores:
            record["svd_score"] = svd_scores.get((uid, pid), 0.0)

        # Etkileşim skoru (proxy feature)
        record["interaction_score"] = row["interaction_score"]

        records.append(record)

    df = pd.DataFrame(records).fillna(0)
    pos_rate = df["label"].mean()
    logger.info(
        f"[LGBM] DataFrame hazır: {len(df)} satır | "
        f"pozitif oran: {pos_rate:.3f}"
    )
    return df


class LGBMTrainer:
    """
    LightGBM ranking modeli eğitim sınıfı.

    lightgbm yüklü değilse LogisticRegression fallback kullanır.
    """

    def __init__(self, config: LGBMConfig = None):
        self.config = config or LGBMConfig()

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> LGBMResult:
        """
        Eğitim DataFrame'inden LightGBM modeli eğitir.

        Args:
            train_df : build_training_dataframe() çıktısı
            val_df   : validation seti (opsiyonel)
        Returns:
            LGBMResult
        """
        # Feature ve label ayır
        drop_cols = ["user_id", "post_id", "label"]
        feature_cols = [c for c in train_df.columns if c not in drop_cols]

        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df["label"].values.astype(np.int32)

        X_val, y_val = None, None
        if val_df is not None:
            X_val = val_df[feature_cols].values.astype(np.float32)
            y_val = val_df["label"].values.astype(np.int32)

        logger.info(
            f"[LGBM] Eğitim: {X_train.shape} | "
            f"pozitif: {y_train.sum()} ({y_train.mean():.3f})"
        )

        t0 = time.time()

        if LGB_AVAILABLE:
            model, train_auc, val_auc = self._train_lgbm(
                X_train, y_train, X_val, y_val, feature_cols
            )
            importances = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
        else:
            model, train_auc, val_auc = self._train_fallback(
                X_train, y_train, X_val, y_val
            )
            importances = {f: 1.0 / len(feature_cols) for f in feature_cols}

        elapsed = time.time() - t0
        logger.info(
            f"[LGBM] Tamamlandı — {elapsed:.1f}s | "
            f"train_auc={train_auc:.4f} | val_auc={val_auc}"
        )

        return LGBMResult(
            model=model,
            feature_names=feature_cols,
            feature_importances=importances,
            train_auc=train_auc,
            val_auc=val_auc,
            train_time_s=elapsed,
            config=self.config,
            n_train_samples=len(X_train),
        )

    def _train_lgbm(self, X_train, y_train, X_val, y_val, feature_cols):
        cfg = self.config
        params = {
            "objective":        cfg.objective,
            "metric":           cfg.metric,
            "num_leaves":       cfg.num_leaves,
            "learning_rate":    cfg.learning_rate,
            "min_child_samples": cfg.min_child_samples,
            "subsample":        cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "reg_alpha":        cfg.reg_alpha,
            "reg_lambda":       cfg.reg_lambda,
            "random_state":     cfg.random_state,
            "verbose":          cfg.verbose,
        }

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        dval   = lgb.Dataset(X_val, label=y_val) if X_val is not None else None

        callbacks = [lgb.log_evaluation(period=50)]
        if dval:
            callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds, verbose=False))

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=cfg.n_estimators,
            valid_sets=[dtrain, dval] if dval else [dtrain],
            valid_names=["train", "val"] if dval else ["train"],
            callbacks=callbacks,
        )

        train_auc = model.best_score.get("train", {}).get("auc", 0.0)
        val_auc   = model.best_score.get("val",   {}).get("auc") if dval else None
        return model, train_auc, val_auc

    def _train_fallback(self, X_train, y_train, X_val, y_val):
        """lightgbm yoksa sklearn LogisticRegression kullan."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        model = LogisticRegression(max_iter=500, random_state=self.config.random_state)
        model.fit(X_scaled, y_train)
        model._scaler = scaler   # scaler'ı modele ekle

        train_auc = roc_auc_score(y_train, model.predict_proba(X_scaled)[:, 1])
        val_auc = None
        if X_val is not None and y_val is not None:
            X_val_scaled = scaler.transform(X_val)
            val_auc = round(roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1]), 4)

        logger.info("[LGBM] Fallback: LogisticRegression kullanıldı")
        return model, round(train_auc, 4), val_auc

    def predict(
        self,
        result: LGBMResult,
        feature_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Tahmin skoru üretir.

        Args:
            result    : LGBMResult
            feature_df: feature kolonları içeren DataFrame (label olmadan)
        Returns:
            probabilities: shape (n,) — tıklama olasılığı [0, 1]
        """
        X = feature_df[result.feature_names].values.astype(np.float32)

        if LGB_AVAILABLE and isinstance(result.model, lgb.Booster):
            return result.model.predict(X)
        else:
            # Fallback model
            scaler = getattr(result.model, "_scaler", None)
            X_s = scaler.transform(X) if scaler else X
            return result.model.predict_proba(X_s)[:, 1]

    def recommend(
        self,
        result: LGBMResult,
        feature_set: FeatureSet,
        user_id: str,
        als_scores: Optional[dict] = None,
        svd_scores: Optional[dict] = None,
        top_k: int = 20,
    ) -> list[dict]:
        """
        Eğitilmiş LightGBM modeliyle öneri üretir.

        Kullanıcının görmediği tüm postlar için feature vektörü
        oluşturur ve tıklama olasılığını tahmin eder.
        """
        if user_id not in feature_set.user_index:
            logger.warning(f"[LGBM] Kullanıcı bulunamadı: {user_id[:8]}")
            return []

        u_idx = feature_set.user_index[user_id]
        seen_item_indices = set(feature_set.user_item_matrix[u_idx].nonzero()[1].tolist())
        seen_post_ids = {feature_set.index_item[i] for i in seen_item_indices}

        # Görülmemiş postlar için feature vektörü
        candidate_records = []
        candidate_post_ids = []

        user_feats = feature_set.user_features
        post_feats = feature_set.post_features

        u = user_feats.loc[user_id] if user_id in user_feats.index else pd.Series(dtype=float)

        for pid, p_idx in feature_set.item_index.items():
            if pid in seen_post_ids:
                continue

            record = {
                "u_total_interactions": u.get("total_interactions", 0),
                "u_like_rate":          u.get("like_rate", 0),
                "u_share_rate":         u.get("share_rate", 0),
                "u_save_rate":          u.get("save_rate", 0),
                "u_avg_dwell_ms":       u.get("avg_dwell_ms", 0),
                "u_avg_scroll":         u.get("avg_scroll", 0),
                "u_recency_days":       u.get("recency_days", 0),
                "u_log_interactions":   u.get("log_interactions", 0),
            }

            if pid in post_feats.index:
                p = post_feats.loc[pid]
                record.update({
                    "p_unique_users":     p.get("unique_users", 0),
                    "p_popularity_score": p.get("popularity_score", 0),
                    "p_like_rate":        p.get("like_rate", 0),
                    "p_share_rate":       p.get("share_rate", 0),
                    "p_avg_dwell_ms":     p.get("avg_dwell_ms", 0),
                    "p_freshness":        p.get("freshness", 0),
                    "p_post_age_days":    p.get("post_age_days", 0),
                    "p_log_unique_users": p.get("log_unique_users", 0),
                })

            if als_scores:
                record["als_score"] = als_scores.get((user_id, pid), 0.0)
            if svd_scores:
                record["svd_score"] = svd_scores.get((user_id, pid), 0.0)

            record["interaction_score"] = 0.0  # henüz etkileşim yok

            candidate_records.append(record)
            candidate_post_ids.append(pid)

        if not candidate_records:
            return []

        feature_df = pd.DataFrame(candidate_records).fillna(0)
        scores = self.predict(result, feature_df)

        # Sırala
        indexed = list(zip(candidate_post_ids, scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (post_id, score) in enumerate(indexed[:top_k], 1):
            results.append({
                "post_id": post_id,
                "score":   round(float(score), 4),
                "rank":    rank,
                "source":  "lgbm",
            })

        logger.info(f"[LGBM] {len(results)} öneri — user={user_id[:8]}")
        return results
