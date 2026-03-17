"""
config/database.py
==================
Khởi tạo và quản lý kết nối database với SQLAlchemy.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from loguru import logger

Base = declarative_base()
_Session = None


def init_database(database_url: str):
    """Khởi tạo engine và session factory."""
    global _Session

    try:
        engine = create_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
        )
        _Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        # Tạo tất cả bảng từ ORM models
        from database.models import (  # noqa: F401 – import để register models
            CustomerRecord, VisitRecord, ZoneEvent, AlertRecord, ModelMetric
        )
        Base.metadata.create_all(bind=engine)
        logger.info(f"✅ Database kết nối thành công: {database_url}")
    except Exception as e:
        logger.warning(f"⚠️ Không thể kết nối database: {e}. Chạy ở chế độ không lưu dữ liệu.")


def get_db_session():
    """
    Cung cấp database session theo dạng context manager.

    Usage:
        with get_db_session() as session:
            session.add(record)
    """
    if _Session is None:
        raise RuntimeError("Database chưa được khởi tạo. Gọi init_database() trước.")

    session = _Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
