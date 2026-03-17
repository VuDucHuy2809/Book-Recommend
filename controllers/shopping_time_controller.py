"""
controllers/shopping_time_controller.py
========================================
CONTROLLER – Quản lý và tổng hợp thời gian mua sắm của khách hàng.
"""

from typing import Dict, List

from loguru import logger

from models.shopping_time import ShoppingTimeModel, CustomerSession
from views.shopping_time_view import ShoppingTimeView
from config.settings import Settings


class ShoppingTimeController:
    """Điều phối tracking thời gian mua sắm theo từng camera."""

    def __init__(self, settings: Settings):
        self.settings = settings
        # Mỗi camera có model riêng
        self._models: Dict[str, ShoppingTimeModel] = {}
        self._view = ShoppingTimeView()
        logger.info("✅ ShoppingTimeController khởi tạo.")

    def _get_model(self, cam_id: str) -> ShoppingTimeModel:
        if cam_id not in self._models:
            self._models[cam_id] = ShoppingTimeModel()
        return self._models[cam_id]

    def process(self, detections: List, cam_id: str) -> Dict:
        """Cập nhật sessions và render thống kê lên frame."""
        model = self._get_model(cam_id)
        ids = [d.customer_id for d in detections]
        model.update(ids, cam_id)
        stats = model.get_stats()
        logger.debug(f"[{cam_id}] Avg shopping time: {stats.get('avg_seconds', 0):.0f}s")
        return stats

    def get_session(self, cam_id: str, customer_id: int) -> CustomerSession:
        return self._get_model(cam_id).get_session(customer_id)

    def get_stats(self, cam_id: str) -> Dict:
        return self._get_model(cam_id).get_stats()
