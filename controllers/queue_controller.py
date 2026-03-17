"""
controllers/queue_controller.py
=================================
CONTROLLER – Quản lý hàng chờ và gửi cảnh báo khi hàng quá dài.
"""

from typing import Dict, List

from loguru import logger

from models.queue_management import QueueManagementModel
from views.queue_view import QueueView
from config.settings import Settings


class QueueController:
    def __init__(self, settings: Settings, alert_service=None):
        self.settings = settings
        self.alert_service = alert_service

        queue_zone_ids = [
            z["id"] for z in settings.ZONE_DEFINITIONS
            if "cashier" in z["id"].lower() or "queue" in z["id"].lower()
        ]
        self._model = QueueManagementModel(queue_zone_ids)
        self._view = QueueView()
        self._alert_cooldown: Dict[str, float] = {}
        logger.info(f"✅ QueueController khởi tạo. Queue zones: {queue_zone_ids}")

    def process(self, detections: List, frame, cam_id: str,
                zone_occupancy: Dict[str, List[int]] = None) -> Dict:
        """Cập nhật hàng chờ, render và gửi alert nếu cần."""
        zone_occupancy = zone_occupancy or {}
        status = self._model.update(zone_occupancy)
        self._view.render(frame, status)

        # Gửi alert nếu hàng quá dài
        if self._model.is_overcrowded(self.settings.QUEUE_ALERT_THRESHOLD):
            logger.warning(
                f"[{cam_id}] 🚨 Hàng chờ dài: {status['current_queue_length']} người!"
            )
            if self.alert_service:
                self.alert_service.send(
                    title="Hàng chờ quá dài",
                    message=f"Quầy {cam_id}: {status['current_queue_length']} người chờ. "
                            f"Avg wait: {status['avg_wait_seconds']}s",
                    severity="WARNING"
                )

        return status

    def get_status(self) -> Dict:
        return self._model.get_status()
