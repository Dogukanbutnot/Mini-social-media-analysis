from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqlalchemy.pool import QueuePool

from .config import DatabaseConfig, db_config


class Base(DeclarativeBase):
    """Tüm ORM modellerinin türeyeceği temel sınıf."""
    pass


class Database:
    """
    SQLAlchemy engine ve session yönetimi.

    Kullanım:
        db = Database(config)
        with db.session() as session:
            users = session.query(User).all()
    """

    def __init__(self, config: DatabaseConfig = db_config):
        self.config = config
        self._engine = create_engine(
            config.url,
            poolclass=QueuePool,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_pre_ping=True,   # bağlantı kopuklarını otomatik tespit et
            echo=config.echo,
        )
        self._SessionFactory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,  # commit sonrası objeleri kullanmaya devam et
        )
        self._register_listeners()

    def _register_listeners(self):
        """pgvector ve pg_trgm extension'larını her yeni bağlantıda aktifleştir."""
        @event.listens_for(self._engine, "connect")
        def on_connect(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("SET search_path TO public")
            cursor.close()

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Context manager ile session yönetimi.
        Hata durumunda otomatik rollback yapar.
        """
        session = self._SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def health_check(self) -> bool:
        """Veritabanı bağlantısını test eder."""
        try:
            with self.session() as s:
                s.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def create_all_tables(self):
        """Tüm tabloları oluşturur (geliştirme ortamı için)."""
        Base.metadata.create_all(self._engine)

    def drop_all_tables(self):
        """Tüm tabloları siler (dikkatli kullan!)."""
        Base.metadata.drop_all(self._engine)

    @property
    def engine(self):
        return self._engine


# Global database instance
database = Database()
