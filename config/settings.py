"""
config/settings.py
==================
Cấu hình toàn cục hệ thống Camera AI Fahasa.
Đọc từ biến môi trường (.env) với giá trị mặc định hợp lý.
"""

import json
import os
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Singleton cấu hình toàn hệ thống."""

    # ── App ──────────────────────────────────────────────────────────────
    APP_NAME: str = os.getenv("APP_NAME", "FahasaAICamera")
    APP_ENV: str  = os.getenv("APP_ENV", "development")
    DEBUG: bool   = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Database ─────────────────────────────────────────────────────────
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite:///fahasa_ai.db"
    )
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # ── Camera Sources ───────────────────────────────────────────────────
    # Format: list of dicts {"id": str, "name": str, "source": str|int}
    # Mặc định: 1 webcam USB nếu chưa cấu hình
    CAMERA_SOURCES: List[Dict[str, Any]] = json.loads(
        os.getenv(
            "CAMERA_SOURCES",
            '[{"id":"cam_001","name":"Default Camera","source":0}]'
        )
    )

    # ── AI Models ────────────────────────────────────────────────────────
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")
    FACE_ANALYZER: str   = os.getenv("FACE_MODEL", "deepface")   # deepface | insightface
    TRACKER_TYPE: str    = os.getenv("TRACKER", "bytetrack")     # bytetrack | botsort

    # ── Thresholds ───────────────────────────────────────────────────────
    DETECTION_CONFIDENCE: float = float(os.getenv("DETECTION_CONFIDENCE", "0.5"))
    ANOMALY_CONFIDENCE: float   = float(os.getenv("ANOMALY_CONFIDENCE_THRESHOLD", "0.75"))
    QUEUE_ALERT_THRESHOLD: int  = int(os.getenv("QUEUE_ALERT_THRESHOLD", "5"))
    ZONE_DWELL_ALERT_SECONDS: int = int(os.getenv("ZONE_DWELL_ALERT_SECONDS", "300"))

    # ── Zone Definitions ─────────────────────────────────────────────────
    # Mỗi zone: {"id": str, "name": str, "polygon": [[x,y], ...]}
    ZONE_DEFINITIONS: List[Dict[str, Any]] = [
        {"id": "z_entrance", "name": "Cửa vào / Ra",       "polygon": [[0,0],[200,0],[200,400],[0,400]]},
        {"id": "z_children", "name": "Sách thiếu nhi",     "polygon": [[200,0],[500,0],[500,400],[200,400]]},
        {"id": "z_novel",    "name": "Tiểu thuyết / Văn học","polygon": [[500,0],[800,0],[800,400],[500,400]]},
        {"id": "z_cashier",  "name": "Quầy thanh toán",    "polygon": [[800,0],[1080,0],[1080,400],[800,400]]},
    ]

    # ── MLflow ───────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "fahasa-camera-ai")

    # ── Alerts ───────────────────────────────────────────────────────────
    ALERT_WEBHOOK_URL: str = os.getenv("ALERT_WEBHOOK_URL", "")
    ALERT_EMAIL: str       = os.getenv("ALERT_EMAIL", "")

    # ── Frame Processing ─────────────────────────────────────────────────
    FRAME_WIDTH: int  = int(os.getenv("FRAME_WIDTH", "1280"))
    FRAME_HEIGHT: int = int(os.getenv("FRAME_HEIGHT", "720"))
    FRAME_SKIP: int   = int(os.getenv("FRAME_SKIP", "2"))   # xử lý 1/N frame để tiết kiệm CPU

    def __repr__(self) -> str:
        return (
            f"Settings(env={self.APP_ENV}, cameras={len(self.CAMERA_SOURCES)}, "
            f"model={self.YOLO_MODEL_PATH}, tracker={self.TRACKER_TYPE})"
        )
