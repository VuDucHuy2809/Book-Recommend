"""
controllers/monitor_controller.py
====================================
CONTROLLER – MLOps: điều phối theo dõi hiệu suất mô hình.
"""

import time
from typing import Dict

from loguru import logger

from models.model_monitor import ModelMonitorModel, ModelHealthReport
from views.monitor_view import MonitorView
from config.settings import Settings


class MonitorController:
    def __init__(self, settings: Settings, alert_service=None):
        self.settings = settings
        self.alert_service = alert_service
        self._model = ModelMonitorModel(model_name="YOLOv11")
        self._view = MonitorView()
        self._last_report_time = time.time()
        self._report_interval = 60  # Giây giữa các lần báo cáo
        logger.info("✅ MonitorController khởi tạo.")

    def log_frame_metrics(
        self,
        cam_id: str,
        inference_ms: float = 0.0,
        detection_count: int = 0,
        anomaly_count: int = 0
    ):
        """Ghi lại metrics của frame vừa xử lý."""
        self._model.log_frame(cam_id, inference_ms, detection_count, anomaly_count)

        # Báo cáo định kỳ
        if time.time() - self._last_report_time >= self._report_interval:
            self._report()
            self._last_report_time = time.time()

    def _report(self):
        report = self._model.get_health_report()
        logger.info(
            f"📊 Model Health | FPS={report.avg_fps:.1f} | "
            f"Latency={report.avg_latency_ms:.0f}ms | "
            f"Drift={report.drift_score:.2f} | {report.recommendation}"
        )
        if report.recommendation == "RETRAIN" and self.alert_service:
            self.alert_service.send(
                title="🔁 Model Drift Detected",
                message=f"YOLOv11 drift score={report.drift_score}. "
                        f"FPS={report.avg_fps}. Đề xuất retrain mô hình.",
                severity="WARNING"
            )
        self._model.export_to_mlflow()

    def get_health_report(self) -> ModelHealthReport:
        return self._model.get_health_report()
