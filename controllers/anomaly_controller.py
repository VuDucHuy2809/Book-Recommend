"""
controllers/anomaly_controller.py
===================================
CONTROLLER – Điều phối phát hiện hành vi bất thường và gửi cảnh báo.
"""

from typing import Dict, List

from loguru import logger

from models.anomaly_detection import AnomalyDetectionModel, AnomalyEvent
from views.anomaly_view import AnomalyView
from config.settings import Settings


class AnomalyController:
    def __init__(self, settings: Settings, alert_service=None):
        self.settings = settings
        self.alert_service = alert_service
        self._model = AnomalyDetectionModel()
        self._view = AnomalyView()
        self._event_log: List[AnomalyEvent] = []
        logger.info("✅ AnomalyController khởi tạo.")

    def process(
        self, detections: List, frame, cam_id: str,
        zone_occupancy: Dict[str, List[int]] = None
    ) -> List[AnomalyEvent]:
        """Phát hiện bất thường, render cảnh báo và gửi alert."""
        zone_occupancy = zone_occupancy or {}
        events = self._model.detect(detections, zone_occupancy, cam_id)

        if events:
            self._view.render(frame, events, detections)
            self._event_log.extend(events)

            for event in events:
                logger.warning(
                    f"[{cam_id}] 🔴 ANOMALY: {event.anomaly_type.value} | "
                    f"{event.description} (conf={event.confidence:.2f})"
                )
                if self.alert_service and event.confidence >= self.settings.ANOMALY_CONFIDENCE:
                    self.alert_service.send(
                        title=f"⚠️ {event.anomaly_type.value.upper()}",
                        message=event.description,
                        severity="CRITICAL"
                    )

        return events

    def get_recent_events(self, limit: int = 20) -> List[Dict]:
        return [e.to_dict() for e in self._event_log[-limit:]]
