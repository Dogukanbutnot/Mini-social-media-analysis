# 📊 Social Media Analytics Backend

> **AI Engineer odaklı**, üretim kalitesinde sosyal medya analitik + ML altyapısı.  
> PostgreSQL + SQLAlchemy ORM + Gerçek ML Pipeline ile inşa edilmiştir.

---

## 🗂️ İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Proje Yapısı](#-proje-yapısı)
- [Dosya Açıklamaları](#-dosya-açıklamaları)
- [Veritabanı Şeması](#-veritabanı-şeması)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [ML Pipeline](#-ml-pipeline)
- [Servisler](#-servisler)
- [Teknolojiler](#-teknolojiler)
- [Yol Haritası](#-yol-haritası)

---

## 🎯 Proje Hakkında

Bu proje, sosyal medya platformlarının arka tarafında çalışan **analitik ve öneri sistemi altyapısını** sıfırdan tasarlamak amacıyla geliştirilmiştir. Standart bir CRUD backend'in ötesine geçerek şu alanlara odaklanır:

- **Recommendation Engine** — User-based CF (Pearson), Item-based CF (Cosine), Content-based Filtering, Hybrid ve Cold Start stratejileri
- **Analitik** — Günlük engagement analizi, rolling window, cohort retention, churn tespiti, network etki skoru
- **ML Pipeline** — Feature engineering, sentence-transformers embedding, ALS / SVD / LightGBM eğitimi, NDCG & Precision@K karşılaştırması
- **A/B Test & Model Monitoring** — CTR karşılaştırması, istatistiksel anlamlılık (z-testi), batch pipeline

### Neden Bu Proje?

Çoğu SQL/Python eğitimi temel CRUD örneklerinde kalır. Bu proje şunları bir arada gösterir:

| Katman | Kapsam |
|--------|--------|
| SQL | Window functions, CTE, Pearson korelasyonu, cohort analizi, IVFFlat index, Partial index |
| Python | SQLAlchemy 2.0 ORM, Repository pattern, Service layer, dataclass'lar |
| ML | ALS, SVD, LightGBM, sentence-transformers embedding, NDCG / Precision@K / Coverage |
| DevOps | Batch pipeline, model versiyonlama, A/B test istatistiği |

---

## 📁 Proje Yapısı

```
social_media_analytics/
│
├── core/
│   ├── config.py                   # Yapılandırma dataclass'ları
│   └── database.py                 # SQLAlchemy engine & session yönetimi
│
├── models/
│   └── orm.py                      # 8 adet SQLAlchemy ORM modeli
│
├── repositories/
│   └── repo.py                     # Veritabanı erişim katmanı (5 repository)
│
├── services/
│   ├── recommendation.py           # Öneri motoru (CF + CBF + Hybrid + Cold Start)
│   ├── analytics.py                # Engagement, cohort, churn, network analizi
│   └── ab_test.py                  # A/B test & model monitoring
│
├── ml/
│   ├── data/
│   │   ├── feature_engineering.py  # Sparse matris, user/item feature vektörleri
│   │   └── embeddings.py           # sentence-transformers embedding üretimi
│   ├── models/
│   │   ├── als_model.py            # ALS — implicit feedback CF
│   │   ├── svd_model.py            # SVD — matris faktörizasyonu
│   │   └── lgbm_model.py           # LightGBM — feature-based ranking
│   ├── evaluation/
│   │   └── metrics.py              # NDCG, Precision@K, Recall@K, Coverage, Novelty
│   └── pipeline/
│       └── run_pipeline.py         # Uçtan uca pipeline orkestrasyonu
│
├── main.py                         # Mock veriyle çalışan demo runner
└── requirements.txt                # Python bağımlılıkları
```

---

## 📄 Dosya Açıklamaları

### `core/config.py`
Tüm yapılandırmaları `dataclass` olarak tutar. Üç ana config sınıfı içerir:

- **`DatabaseConfig`** — PostgreSQL bağlantı bilgileri, pool ayarları. `from_env()` metodu ile `.env` dosyasından otomatik okur.
- **`RecommendationConfig`** — CF parametreleri (komşu sayısı, minimum ortak etkileşim), hybrid ağırlıklar, sinyal ağırlıkları, cold start ayarları.
- **`AnalyticsConfig`** — Rolling window, cohort eşikleri, network analizi ağırlıkları.

---

### `core/database.py`
SQLAlchemy engine ve session yönetimini kapsüller.

- **`Database` sınıfı** — `QueuePool` ile connection pooling, `pool_pre_ping=True` ile kopuk bağlantı tespiti.
- **`session()` context manager** — `with db.session() as s:` kullanımı; hata durumunda otomatik rollback.
- **`health_check()`** — Bağlantı canlılık testi.

---

### `models/orm.py`
SQLAlchemy 2.0 `mapped_column` syntax ile yazılmış 8 ORM modeli:

| Model | Tablo | Açıklama |
|-------|-------|----------|
| `User` | `users` | Kullanıcı profili + `VECTOR(384)` embedding |
| `Post` | `posts` | İçerik + virality/sentiment skoru + içerik embedding |
| `Follow` | `follows` | Takip ilişkisi + `interaction_weight` (GNN edge feature) |
| `Interaction` | `interactions` | Etkileşimler + `dwell_time_ms` + `scroll_depth` |
| `Hashtag` | `hashtags` | Hashtag + `trend_score` |
| `PostHashtag` | `post_hashtags` | M:N köprü tablo |
| `UserSession` | `user_sessions` | Oturum verisi (churn modeli için) |
| `Recommendation` | `recommendations` | Model çıktıları + `was_clicked` feedback |

Her model ilgili index tanımlarını `__table_args__` içinde barındırır.

---

### `repositories/repo.py`
Tüm SQL sorgularını servis katmanından soyutlayan 5 repository sınıfı:

- **`UserRepository`** — Kullanıcı sorgulama, aktif kullanıcı listesi, engagement rate güncelleme.
- **`PostRepository`** — Post sorgulama, viral sıralama, kullanıcının görmediği postlar.
- **`InteractionRepository`** — Etkileşim oluşturma, user-item matrisi ham verisi.
- **`FollowRepository`** — Takip CRUD, interaction weight güncelleme.
- **`RecommendationRepository`** — Toplu kayıt (`bulk_save`), stale temizleme, model istatistikleri, `Precision@K`.

---

### `services/recommendation.py`
Projenin SQL katmanındaki kalbi. 5 öneri stratejisi + batch pipeline:

#### `user_cf(user_id, limit)` — User-Based Collaborative Filtering
Pearson korelasyonu ile kullanıcı benzerliği hesaplar. Hedef kullanıcının takip ettiği kişiler arasından en benzer K komşuyu bulur, onların beğendiği henüz görülmemiş postları ağırlıklı tahmin skoru ile sıralar.

#### `content_based(user_id, limit)` — Content-Based Filtering
Kullanıcının son 90 günlük hashtag kullanım frekansından affinity profili çıkarır. Görülmemiş adayları `hashtag_affinity × 0.40 + freshness × 0.30 + virality × 0.30` formülü ile skorlar. Tazelik, 24 saatlik yarı-ömürlü üstel decay ile hesaplanır.

#### `hybrid(user_id, limit)` — Hybrid Öneri
CF ve CBF skorlarını min-max normalize edip `CF×0.50 + CBF×0.35 + Popularity×0.15` ağırlıkları ile birleştirir. Cold start tespiti yapar; CF verisi yoksa ağırlık otomatik CBF'e kayar.

#### `cold_start(limit)` — Soğuk Başlangıç
Yeni kullanıcılar için her medya tipinden en viral 5 içeriği getirir (diversity zorlanmış). Son 72 saatin trending postları baz alınır.

#### `run_batch_pipeline(model_version)` — Toplu Öneri Üretimi
Tüm aktif kullanıcılar için günlük öneri batch'i çalıştırır. Stale önerileri temizler, kota kontrolü yapar, hataları loglayarak devam eder.

---

### `services/analytics.py`
4 analitik modül:

#### `daily_engagement(since_days, user_id)` — Günlük Engagement
SQL window function ile 7 günlük rolling ortalama hesaplar. `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` penceresi kullanır. Sonuçlar ML time-series feature olarak kullanılabilir.

#### `cohort_retention()` — Cohort Analizi
Kullanıcıları kayıt ayına göre gruplar; ilerleyen aylarda kaç tanesi aktif kalmış hesaplar. Klasik SaaS retention tablosu üretir.

#### `churn_risk_scores(label_filter)` — Churn Tespiti
Aktivite düşüşü (`sessions_last_30d / sessions_prev_30d`) ve pasiflik süresi sinyallerini birleştiren risk skoru. `churn_label` kolonu supervised learning için hazır label: `churned | at_risk | declining | healthy`.

#### `influence_scores(top_n)` — Network Etki Analizi
2-hop PageRank benzeri etki skoru. Doğrudan takipçi ağırlığı × 0.7 + 2. derece bağlantı × 0.3. GNN edge feature'ı olarak kullanılabilir.

#### `ml_feature_store(user_ids)` — ML Feature Store
Model eğitimi için hazır feature vektörü. `log_followers`, `log_posts` gibi log-transform uygulanmış kolonlar dahil. Çıktı doğrudan `pandas.read_sql()` ile DataFrame'e alınabilir.

---

### `services/ab_test.py`
Model performansı ve istatistiksel karşılaştırma:

#### `model_performance(since_days)` — Model Metrik Tablosu
Her model versiyonu için CTR, ortalama skor, tıklama sonrası virality ve dwell time metriklerini hesaplar.

#### `ab_compare(version_a, version_b)` — İstatistiksel A/B Testi
İki model arasında proporsiyon z-testi uygular. `p < 0.05` eşiğinde kazananı ilan eder. Z-skordan p-değeri için Abramowitz & Stegun yaklaşımı kullanır.

#### `precision_at_k(k, since_days)` — Precision@K
Son N günde verilen önerilerin ilk K sırasında kaçı tıklandı? Standart IR metriği.

#### `monitoring_report(since_days)` — Monitoring Dashboard
Tüm metrikleri birleştiren özet rapor. Celery/APScheduler ile günlük Slack bildirimi için kullanılabilir.

---

### `ml/data/feature_engineering.py`
Ham etkileşim verisini ML'e hazır yapılara dönüştürür.

- **`prepare_features(interaction_records)`** — Ana giriş noktası. Etkileşim listesini alır; sparse matris, user feature DataFrame ve post feature DataFrame üretir.
- **`build_user_item_matrix()`** — `(n_users × n_items)` scipy sparse matris. ALS ve SVD'ye doğrudan girdi.
- **`build_user_features()`** — 18 kolonlu kullanıcı feature vektörü: `like_rate`, `share_rate`, `avg_dwell_ms`, `recency_days`, `log_interactions` vb.
- **`build_post_features()`** — 18 kolonlu içerik feature vektörü: `popularity_score`, `freshness`, `post_age_days`, `log_unique_users` vb.
- **`compute_interaction_score()`** — Explicit + implicit sinyalleri `share×5 + save×4 + like×3 + dwell_bonus` formülüyle tek skora indirger.

---

### `ml/data/embeddings.py`
sentence-transformers ile içerik ve kullanıcı embedding'leri üretir.

- **`EmbeddingProducer`** — `all-MiniLM-L6-v2` modeli ile 384 boyutlu vektör üretir. Kütüphane yüklü değilse hash tabanlı deterministik mock embedding kullanır (test ortamı için).
- **`EmbeddingRecommender`** — Post embedding'lerini in-memory index'e alır, kullanıcının beğendiği postların ağırlıklı ortalamasından ilgi vektörü (centroid) üretir, cosine similarity ile öneri sıralar.
- **`compute_user_embedding()`** — Beğenilen post vektörlerinin sinyal ağırlıklı ortalaması. L2 normalize edilmiş.
- **`find_similar()`** — Query embedding'e ANN araması yapar. Görülmüş içerikler otomatik hariç tutulur.

---

### `ml/models/als_model.py`
ALS (Alternating Least Squares) — implicit feedback matris faktörizasyonu.

- `implicit` kütüphanesi varsa BM25 ağırlıklandırmalı production ALS çalışır; yoksa saf NumPy fallback devreye girer.
- **`ALSConfig`** — `factors=64`, `iterations=20`, `regularization=0.01`, `use_bm25=True` varsayılan değerleri.
- **`ALSResult.recommend_for_user()`** — Kullanıcı faktör vektörü ile item faktör matrisinin dot product'ı; görülmüş itemler `-inf` ile maskelenir.

---

### `ml/models/svd_model.py`
SVD (Truncated Singular Value Decomposition) — sklearn tabanlı matris faktörizasyonu.

- **`SVDTrainer.train()`** — `TruncatedSVD` ile `U × Σ` (kullanıcı) ve `V × Σ` (item) embedding matrisleri üretir. L2 normalize seçeneği var.
- **`SVDResult.get_similar_items()`** — Item-item cosine similarity. "Bunu beğenenler bunları da beğendi" senaryosu için kullanılır.
- Açıklanan varyans oranı loglanır; kaç faktörün yeterli olduğu izlenebilir.

---

### `ml/models/lgbm_model.py`
LightGBM — feature tabanlı learning-to-rank modeli.

- **`build_training_dataframe()`** — Her `(user, item)` çifti için kullanıcı + post feature'larını + ALS/SVD stacking skorlarını birleştirir. Label: `interaction_score >= 3.0` (like veya üzeri).
- **`LGBMTrainer.train()`** — Binary classification (`objective=binary`, `metric=auc`). `lightgbm` yoksa `LogisticRegression` fallback.
- **`LGBMTrainer.recommend()`** — Görülmemiş tüm postlar için feature vektörü oluşturur, tıklama olasılığını tahmin eder.
- Feature importance çıktısı hangi sinyalin model kararında ne kadar belirleyici olduğunu gösterir.

---

### `ml/evaluation/metrics.py`
Öneri modellerini karşılaştıran standart IR metrikleri.

- **`precision_at_k()`** — İlk K önerideki ilgili içerik oranı.
- **`recall_at_k()`** — İlgili içeriklerin kaçı ilk K içinde yakalandı.
- **`ndcg_at_k()`** — Normalized Discounted Cumulative Gain. Sıralama kalitesini ölçer; mükemmel sıralama = 1.0.
- **`ModelEvaluator.evaluate()`** — Tüm metrikleri hesaplar, coverage (katalog kapsama oranı) ve novelty (popüler olmayan item önerme oranı) ekler.
- **`ModelEvaluator.train_test_split()`** — Zaman tabanlı split: her kullanıcının son %20 etkileşimi test seti. Gelecekteki davranışı simüle eder.
- **`ModelEvaluator.compare_models()`** — Birden fazla modeli DataFrame olarak yan yana karşılaştırır.

---

### `ml/pipeline/run_pipeline.py`
Tüm ML adımlarını sırayla çalıştıran orkestratör.

Pipeline adımları:
1. Mock veri veya DB verisi ile başlatma
2. Feature engineering (sparse matris + DataFrame'ler)
3. sentence-transformers ile post indexleme + kullanıcı profil üretimi
4. Zaman tabanlı train/test split
5. ALS, SVD, LightGBM ve Embedding-CBF modellerini eğitme
6. Her model için NDCG@5, NDCG@10, P@10, Coverage, Novelty hesaplama
7. Karşılaştırma tablosu + kazanan model ilanı
8. LightGBM feature importance çıktısı

```
python -m social_media_analytics.ml.pipeline.run_pipeline
```

---

### `main.py`
Gerçek PostgreSQL bağlantısı gerektirmeden tüm servislerin çalıştığını gösteren **demo runner**. Mock veri üretir, her servisi çalıştırır ve çıktıları terminale yazar.

---

## 🗄️ Veritabanı Şeması

```
users ──────────────────────────────────────────────────────
  id UUID PK | username | email | embedding VECTOR(384)
  follower_count | following_count | avg_engagement_rate

posts ──────────────────────────────────────────────────────
  id UUID PK | user_id FK | content | media_type
  virality_score | sentiment_score | content_embedding VECTOR(384)

follows ────────────────────────────────────────────────────
  (follower_id, following_id) PK | interaction_weight

interactions ───────────────────────────────────────────────
  id UUID PK | user_id FK | post_id FK | type
  dwell_time_ms | scroll_depth

hashtags + post_hashtags ───────────────────────────────────
  tag | usage_count | trend_score

user_sessions ──────────────────────────────────────────────
  session_duration_s | posts_viewed | device_type

recommendations ────────────────────────────────────────────
  source_user_id FK | target_post_id FK
  score | model_version | was_clicked
```

### Index Stratejisi

| Index Türü | Kullanım Amacı |
|------------|----------------|
| B-Tree | Foreign key join, zaman serisi sıralama, virality/trend sıralama |
| IVFFlat (pgvector) | Embedding ANN araması — kullanıcı ve içerik benzerliği |
| GIN (tsvector) | Full-text search — içerik arama |
| GIN (trigram) | Prefix/infix hashtag araması |
| Partial | `dwell_time_ms > 5000` — sadece yüksek engagement etkileşimleri |

---

## ⚙️ Kurulum

### Gereksinimler

- Python 3.11+
- PostgreSQL 14+ (`pgvector` ve `pg_trgm` extension'ları ile)
- pip

### 1. Repoyu Klonla

```bash
git clone https://github.com/kullanici-adin/social-media-analytics.git
cd social-media-analytics
```

### 2. Sanal Ortam Oluştur

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. PostgreSQL Extension'larını Kur

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

### 4. Ortam Değişkenlerini Ayarla

```bash
cp .env.example .env
```

`.env` dosyasını düzenle:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=social_media_db
DB_USER=postgres
DB_PASSWORD=your_password
DB_ECHO=false
```

### 5. Tabloları Oluştur

```python
from social_media_analytics.core.database import database
database.create_all_tables()
```

---

## 🚀 Kullanım

### Demo — Servis Katmanı (Mock Veri)

PostgreSQL bağlantısı olmadan tüm servisleri test et:

```bash
python -m social_media_analytics.main
```

Çıktıda şunları görürsün:
- Cold start, CF, CBF ve Hybrid önerileri
- Engagement analizi ve churn sıralaması
- Network etki skorları
- A/B test sonuçları

### Demo — ML Pipeline (Mock Veri)

4 modeli eğitip NDCG / Precision@K karşılaştırması yap:

```bash
python -m social_media_analytics.ml.pipeline.run_pipeline
```

Örnek çıktı:
```
               NDCG@5  NDCG@10   P@10  coverage  novelty
ALS            0.0272   0.0422  0.018     0.980   0.378
SVD            0.0163   0.0206  0.012     0.970   0.460
LightGBM       0.0104   0.0238  0.016     0.135   0.464
Embedding-CBF  0.0000   0.0000  0.000     0.000   1.000

🏆 KAZANAN: ALS (NDCG@10=0.0422)
```

### Gerçek Veritabanı ile Kullanım

```python
from social_media_analytics.core.database import database
from social_media_analytics.services.recommendation import RecommendationEngine
from social_media_analytics.services.analytics import AnalyticsService
from social_media_analytics.services.ab_test import ABTestService

# Hybrid öneri üret
with database.session() as session:
    engine = RecommendationEngine(session)
    recs = engine.hybrid(user_id="<uuid>", limit=20)
    for rec in recs:
        print(rec)

# Churn analizi
with database.session() as session:
    analytics = AnalyticsService(session)
    at_risk = analytics.churn_risk_scores(label_filter="at_risk")
    for user in at_risk:
        print(user)

# Batch pipeline (Celery/cron ile tetiklenebilir)
with database.session() as session:
    engine = RecommendationEngine(session)
    stats = engine.run_batch_pipeline(model_version="v2.0")
    print(stats)  # {"processed": 1200, "total_recs": 60000, "errors": 3}
```

---

## 🤖 ML Pipeline

### Mimari

```
Ham Veri (PostgreSQL / mock)
         ↓
Feature Engineering
  ├── scipy sparse matrix  (n_users × n_items)
  ├── user_features_df     (18 kolon)
  └── post_features_df     (18 kolon)
         ↓
Embedding Üretimi
  └── sentence-transformers → 384-dim post vektörleri
      → kullanıcı ilgi profili (ağırlıklı centroid)
         ↓
Train / Test Split  (zaman tabanlı %80/%20)
         ↓
Model Eğitimi
  ├── ALS        (implicit feedback, BM25 ağırlıklı)
  ├── SVD        (TruncatedSVD, L2 normalize)
  ├── LightGBM   (binary CTR tahmini, AUC metriği)
  └── Embedding-CBF (cosine similarity + virality)
         ↓
Değerlendirme
  ├── NDCG@5, NDCG@10
  ├── Precision@K, Recall@K
  ├── Coverage   (katalog kapsama oranı)
  └── Novelty    (popüler olmayan item oranı)
         ↓
Karşılaştırma Tablosu + Kazanan Model
```

### Sinyal Ağırlıkları

| Etkileşim | Ağırlık | Açıklama |
|-----------|---------|----------|
| share | 5.0 | En güçlü pozitif sinyal |
| save | 4.0 | Yüksek ilgi göstergesi |
| like | 3.0 | Standart pozitif sinyal |
| comment | 2.5 | Aktif katılım |
| view | 1.0 + dwell_bonus | Pasif sinyal; dwell time ile güçlenir |
| report | -5.0 | Negatif sinyal |

### sentence-transformers Entegrasyonu

```bash
pip install sentence-transformers
```

```python
from social_media_analytics.ml.data.embeddings import EmbeddingProducer, EmbeddingConfig

producer = EmbeddingProducer(EmbeddingConfig(model_name="all-MiniLM-L6-v2"))

# Tek metin encode
embedding = producer.encode_single("Python ile transformer modeli nasıl fine-tune edilir?")
# → shape: (384,)

# Toplu encode
embeddings = producer.encode(["metin 1", "metin 2", "metin 3"])
# → shape: (3, 384)
```

> `sentence-transformers` yüklü değilse otomatik olarak deterministik mock embedding kullanılır — test ve CI ortamları için uygundur.

### Model Eğitimi

```python
from social_media_analytics.ml.data.feature_engineering import prepare_features
from social_media_analytics.ml.models.als_model import ALSTrainer, ALSConfig
from social_media_analytics.ml.models.svd_model import SVDTrainer, SVDConfig
from social_media_analytics.ml.models.lgbm_model import LGBMTrainer, LGBMConfig, build_training_dataframe

# Feature engineering
feature_set = prepare_features(interaction_records)

# ALS
als_trainer = ALSTrainer(ALSConfig(factors=64, iterations=20))
als_result  = als_trainer.train(feature_set)
recs = als_trainer.recommend(als_result, feature_set, user_id="<uuid>", top_k=20)

# SVD
svd_trainer = SVDTrainer(SVDConfig(n_components=64))
svd_result  = svd_trainer.train(feature_set)
similar = svd_trainer.get_similar_posts(svd_result, feature_set, post_id="<uuid>")

# LightGBM
train_df    = build_training_dataframe(feature_set)
lgbm_trainer = LGBMTrainer(LGBMConfig(n_estimators=200))
lgbm_result  = lgbm_trainer.train(train_df)
recs = lgbm_trainer.recommend(lgbm_result, feature_set, user_id="<uuid>")
```

### Değerlendirme

```python
from social_media_analytics.ml.evaluation.metrics import ModelEvaluator

evaluator = ModelEvaluator(k_values=[5, 10, 20])

# Train/test split
train_df, ground_truth = evaluator.train_test_split(interaction_df, test_ratio=0.2)

# Tek model değerlendir
result = evaluator.evaluate(
    model_name="ALS",
    recommendations={"user_id": ["post_1", "post_2", ...]},
    ground_truth={"user_id": {"post_1", "post_3"}},
    all_item_ids=all_post_ids,
)
print(result)  # NDCG@10=0.0422 | P@10=0.018 | Coverage=0.980

# Birden fazla modeli karşılaştır
df = evaluator.compare_models([als_result, svd_result, lgbm_result])
print(df)
```

---

## 🔧 Servisler

### RecommendationEngine

```python
engine = RecommendationEngine(session)

engine.cold_start(limit=20)
engine.user_cf(user_id, limit=20)
engine.content_based(user_id, limit=20)
engine.hybrid(user_id, limit=20)
engine.persist_recommendations(user_id, recs, model_version="v2.0")
engine.run_batch_pipeline(model_version="v2.0")
```

### AnalyticsService

```python
analytics = AnalyticsService(session)

analytics.daily_engagement(since_days=30)
analytics.hourly_heatmap(since_days=90)
analytics.cohort_retention()
analytics.churn_risk_scores(label_filter="at_risk")
analytics.influence_scores(top_n=100)
analytics.ml_feature_store(user_ids=[...])
```

### ABTestService

```python
ab = ABTestService(session)

ab.model_performance(since_days=14)
ab.ab_compare("v1.0", "v2.0")
ab.precision_at_k(k=10, since_days=7)
ab.monitoring_report(since_days=14)
```

---

## 🛠️ Teknolojiler

| Teknoloji | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| **PostgreSQL** | 14+ | Ana veritabanı |
| **pgvector** | 0.5+ | Embedding vektör araması (IVFFlat ANN) |
| **SQLAlchemy** | 2.0+ | ORM, session yönetimi, connection pooling |
| **psycopg2** | 2.9+ | PostgreSQL Python sürücüsü |
| **numpy** | 1.24+ | Matris işlemleri, embedding hesapları |
| **scipy** | 1.11+ | Sparse matris (CSR format), ALS için |
| **pandas** | 2.0+ | Feature DataFrame'leri, train/test split |
| **scikit-learn** | 1.3+ | TruncatedSVD, normalizasyon, LR fallback |
| **lightgbm** | 4.0+ | Feature-based ranking modeli |
| **sentence-transformers** | 2.2+ | 384-dim içerik ve kullanıcı embedding'leri |
| **implicit** | 0.7+ | Üretim kalitesi ALS (opsiyonel) |
| **Python** | 3.11+ | Tüm servis ve ML katmanı |

### requirements.txt Açıklaması

```
sqlalchemy>=2.0.0       # ORM — mapped_column, DeclarativeBase (2.0 syntax)
psycopg2-binary>=2.9.0  # PostgreSQL sürücüsü — binary paket, derleme gerektirmez
pgvector>=0.2.0         # SQLAlchemy için Vector tip desteği

numpy>=1.24.0           # Matris işlemleri, embedding hesapları
scipy>=1.11.0           # Sparse matris (CSR), ALS için
pandas>=2.0.0           # Feature DataFrame'leri
scikit-learn>=1.3.0     # TruncatedSVD, normalizasyon
lightgbm>=4.0.0         # LightGBM ranking modeli

# Opsiyonel — gerçek embedding için
sentence-transformers>=2.2.0
implicit>=0.7.0         # Üretim kalitesi ALS + BM25
```

---

## 🗺️ Yol Haritası

- [x] PostgreSQL şema + index stratejisi
- [x] SQLAlchemy ORM + Repository + Service katmanları
- [x] Recommendation Engine (CF, CBF, Hybrid, Cold Start)
- [x] Analytics (engagement, cohort, churn, network)
- [x] A/B Test & Model Monitoring
- [x] ML Pipeline (Feature Engineering + Embedding + ALS + SVD + LightGBM + NDCG)
- [ ] FastAPI endpoint katmanı
- [ ] Alembic ile migration sistemi
- [ ] Celery batch pipeline entegrasyonu
- [ ] Grafana dashboard (model monitoring)
- [ ] Docker Compose ile tek komutla ayağa kalkma


---

## 📝 Lisans

MIT License — dilediğiniz gibi kullanabilirsiniz.
