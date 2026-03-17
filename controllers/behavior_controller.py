"""
controllers/behavior_controller.py
=====================================
CONTROLLER – Điều phối phân tích hành vi khách hàng theo từng khu vực.
"""

from typing import Dict, List

from loguru import logger

from models.behavior_analysis import BehaviorAnalysisModel, BehaviorResult
from views.behavior_view import BehaviorView
from config.settings import Settings


class BehaviorController:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._models: Dict[str, BehaviorAnalysisModel] = {}
        self._view = BehaviorView()
        logger.info("✅ BehaviorController khởi tạo.")

    def _get_model(self, cam_id: str) -> BehaviorAnalysisModel:
        if cam_id not in self._models:
            self._models[cam_id] = BehaviorAnalysisModel()
        return self._models[cam_id]

    def process(
        self, detections: List, frame, cam_id: str,
        zone_map: Dict[int, str] = None
    ) -> List[BehaviorResult]:
        """Phân tích hành vi và render lên frame."""
        model = self._get_model(cam_id)
        results = model.analyze(detections, zone_map)
        self._view.render(frame, results)
        return results

    def get_zone_summary(self, cam_id: str, results: List[BehaviorResult]) -> Dict:
        model = self._get_model(cam_id)
        return model.get_zone_behavior_summary(results)
