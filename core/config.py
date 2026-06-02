from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "social_media_db"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False  # SQL loglarını göster

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "social_media_db"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
        )


@dataclass
class RecommendationConfig:
    # Collaborative filtering
    cf_min_common_items: int = 5        # Pearson için minimum ortak etkileşim
    cf_top_k_neighbors: int = 20        # Kaç komşu kullanılsın
    cf_min_neighbor_votes: int = 3      # Güvenilirlik eşiği

    # Content-based filtering
    cbf_embedding_dim: int = 384        # sentence-transformers boyutu
    cbf_interest_window_days: int = 30  # İlgi profili penceresi
    cbf_freshness_half_life_hours: float = 24.0  # Tazelik yarı-ömrü

    # Hybrid ağırlıklar
    hybrid_cf_weight: float = 0.50
    hybrid_cbf_weight: float = 0.35
    hybrid_popularity_weight: float = 0.15

    # Sinyal ağırlıkları (user-item matrix)
    signal_weights: dict = field(default_factory=lambda: {
        "share":   5.0,
        "save":    4.0,
        "like":    3.0,
        "view":    1.0,   # + dwell time bonus
        "report": -5.0,
    })

    # Batch pipeline
    recommendation_ttl_days: int = 7
    daily_recommendation_quota: int = 50
    precision_at_k: int = 10

    # Cold start
    cold_start_per_media_type: int = 5
    cold_start_trending_hours: int = 72


@dataclass
class AnalyticsConfig:
    rolling_window_days: int = 7
    cohort_min_users: int = 10
    churn_lookback_days: int = 30
    influence_hop2_weight: float = 0.3
    jaccard_min_common: int = 3


# Uygulama genelinde kullanılan default config'ler
db_config = DatabaseConfig.from_env()
rec_config = RecommendationConfig()
analytics_config = AnalyticsConfig()
