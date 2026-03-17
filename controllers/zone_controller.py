"""
controllers/zone_controller.py
================================
CONTROLLER – Điều phối phân tích tương tác theo khu vực (zone).
"""

from typing import Dict, List

from loguru import logger

from models.zone_interaction import ZoneInteractionModel
from views.zone_view import ZoneView
from config.settings import Settings


class ZoneController:
    def __init__(self, settings: Settings, alert_service=None):
        self.settings = settings
        self.alert_service = alert_service
        self._models: Dict[str, ZoneInteractionModel] = {}
        self._view = ZoneView(settings.ZONE_DEFINITIONS)
        logger.info("✅ ZoneController khởi tạo.")

    def _get_model(self, cam_id: str, frame_shape: tuple) -> ZoneInteractionModel:
        if cam_id not in self._models:
            self._models[cam_id] = ZoneInteractionModel(
                self.settings.ZONE_DEFINITIONS, frame_shape
            )
        return self._models[cam_id]

    def process(self, detections: List, frame, cam_id: str) -> Dict[str, List[int]]:
        """Cập nhật zone occupancy, heatmap và render."""
        model = self._get_model(cam_id, frame.shape)
        zone_occupancy = model.update(detections)

        # Render zones và heatmap lên frame
        self._view.render(frame, zone_occupancy, model.zones)

        # Alert nếu zone quá tải
        for zone_id, occupants in zone_occupancy.items():
            if len(occupants) >= self.settings.QUEUE_ALERT_THRESHOLD:
                logger.warning(f"[{cam_id}] ⚠️ Zone '{zone_id}' đông: {len(occupants)} người")

        return zone_occupancy

    def get_stats(self, cam_id: str) -> List[Dict]:
        model = self._models.get(cam_id)
        return model.get_zone_stats() if model else []
