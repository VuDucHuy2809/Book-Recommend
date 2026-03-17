"""
controllers/customer_controller.py
===================================
CONTROLLER – Điều phối nhận diện khách hàng, gán Customer ID duy nhất,
lưu database và render view.
"""

import time
from typing import Dict, List, Any

from loguru import logger

from models.customer_detection import CustomerDetectionModel, CustomerData
from views.customer_view import CustomerView
from config.settings import Settings


class CustomerController:
    """
    Điều phối toàn bộ pipeline phát hiện và phân tích khách hàng.

    Mỗi khách hàng được gán một Customer ID duy nhất thông qua ByteTrack.
    ID được duy trì liên tục trong suốt phiên camera.
    """

    def __init__(self, settings: Settings, alert_service=None):
        self.settings = settings
        self.alert_service = alert_service
        self._model = CustomerDetectionModel(
            model_path=settings.YOLO_MODEL_PATH,
            confidence=settings.DETECTION_CONFIDENCE,
            face_analyzer=settings.FACE_ANALYZER,
        )
        self._view = CustomerView()

        # Thống kê: cam_id → { total_in, total_out, peak }
        self._cam_stats: Dict[str, Dict] = {}
        # customer_id đã từng gặp theo cam_id
        self._cam_seen_ids: Dict[str, set] = {}

        logger.info("✅ CustomerController khởi tạo.")

    def process(self, frame, cam_id: str) -> List[CustomerData]:
        """
        Xử lý một frame từ camera: detect → track → phân tích → render.

        Args:
            frame: BGR numpy array từ camera.
            cam_id: ID camera.

        Returns:
            Danh sách CustomerData với Customer ID duy nhất.
        """
        t_start = time.time()

        # ── 1. Detect + Track ────────────────────────────────────────────
        detections: List[CustomerData] = self._model.detect(frame, cam_id)

        # ── 2. Cập nhật thống kê khách mới ──────────────────────────────
        if cam_id not in self._cam_seen_ids:
            self._cam_seen_ids[cam_id] = set()
        if cam_id not in self._cam_stats:
            self._cam_stats[cam_id] = {"total_unique": 0, "current": 0, "peak": 0}

        new_ids = set(d.customer_id for d in detections)
        newly_arrived = new_ids - self._cam_seen_ids[cam_id]
        self._cam_seen_ids[cam_id].update(new_ids)
        self._cam_stats[cam_id]["total_unique"] += len(newly_arrived)
        self._cam_stats[cam_id]["current"] = len(detections)
        self._cam_stats[cam_id]["peak"] = max(
            self._cam_stats[cam_id]["peak"], len(detections)
        )

        # ── 3. Log khách mới ─────────────────────────────────────────────
        for cid in newly_arrived:
            logger.debug(f"[{cam_id}] 🆕 Khách hàng mới: ID #{cid}")

        # ── 4. Render lên frame ──────────────────────────────────────────
        self._view.render(frame, detections, self._cam_stats[cam_id])

        elapsed_ms = (time.time() - t_start) * 1000
        logger.debug(f"[{cam_id}] Customer pipeline: {elapsed_ms:.1f}ms, "
                     f"{len(detections)} người hiện tại.")

        return detections

    def get_stats(self, cam_id: str) -> Dict:
        return self._cam_stats.get(cam_id, {})

    def get_all_stats(self) -> Dict[str, Dict]:
        return dict(self._cam_stats)
