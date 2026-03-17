"""
models/model_monitor.py
=======================
MODEL – MLOps: Thu thập và phân tích metrics hiệu suất mô hình AI.

Theo dõi:
- FPS xử lý mỗi camera
- Inference latency (ms/frame)
- Detection count per frame
- Model accuracy drift (so sánh với ground truth)
- Memory / GPU usage
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque


@dataclass
class FrameMetric:
    """Metrics cho một frame đơn lẻ."""
    cam_id: str
    timestamp: float
    inference_ms: float          # Thời gian YOLO inference
    detection_count: int         # Số lượng người phát hiện được
    fps: float
    anomaly_count: int = 0


@dataclass
class ModelHealthReport:
    """Báo cáo sức khỏe mô hình định kỳ."""
    model_name: str
    avg_fps: float
    avg_latency_ms: float
    total_frames: int
    uptime_seconds: float
    drift_score: float           # 0 = tốt, 1 = cần retrain
    recommendation: str          # "OK" | "MONITOR" | "RETRAIN"


class ModelMonitorModel:
    """
    Thu thập và phân tích metrics hiệu suất mô hình AI.

    Sử dụng sliding window để tính thống kê thời gian thực.
    Tích hợp với MLflow để log metrics.
    """

    WINDOW_SIZE       = 100     # Số frame gần nhất để tính thống kê
    DRIFT_FPS_WARN    = 10.0    # FPS < 10 → cảnh báo
    DRIFT_FPS_RETRAIN = 5.0     # FPS < 5 → đề xuất retrain / scale
    LATENCY_WARN_MS   = 100.0   # Latency > 100ms → cảnh báo

    def __init__(self, model_name: str = "YOLOv11"):
        self.model_name = model_name
        self.start_time = time.time()
        self._metrics_window: Deque[FrameMetric] = deque(maxlen=self.WINDOW_SIZE)
        self._cam_frame_count: Dict[str, int] = {}
        self._cam_last_time: Dict[str, float] = {}

    def log_frame(
        self,
        cam_id: str,
        inference_ms: float,
        detection_count: int,
        anomaly_count: int = 0
    ) -> FrameMetric:
        """
        Ghi lại metrics cho một frame xử lý.

        Args:
            cam_id: ID camera.
            inference_ms: Thời gian inference (ms).
            detection_count: Số người phát hiện.
            anomaly_count: Số anomaly phát hiện.

        Returns:
            FrameMetric vừa ghi.
        """
        now = time.time()

        # Tính FPS
        last = self._cam_last_time.get(cam_id, now)
        fps = 1.0 / (now - last) if (now - last) > 0 else 0
        self._cam_last_time[cam_id] = now
        self._cam_frame_count[cam_id] = self._cam_frame_count.get(cam_id, 0) + 1

        metric = FrameMetric(
            cam_id=cam_id,
            timestamp=now,
            inference_ms=inference_ms,
            detection_count=detection_count,
            fps=fps,
            anomaly_count=anomaly_count
        )
        self._metrics_window.append(metric)
        return metric

    def get_health_report(self) -> ModelHealthReport:
        """Tạo báo cáo sức khỏe mô hình từ metrics gần nhất."""
        if not self._metrics_window:
            return ModelHealthReport(
                model_name=self.model_name,
                avg_fps=0, avg_latency_ms=0,
                total_frames=0,
                uptime_seconds=time.time() - self.start_time,
                drift_score=0, recommendation="OK"
            )

        metrics = list(self._metrics_window)
        avg_fps       = sum(m.fps for m in metrics) / len(metrics)
        avg_latency   = sum(m.inference_ms for m in metrics) / len(metrics)
        total_frames  = sum(self._cam_frame_count.values())

        # Tính drift score
        drift_score = 0.0
        if avg_fps < self.DRIFT_FPS_WARN:
            drift_score += 0.5
        if avg_latency > self.LATENCY_WARN_MS:
            drift_score += 0.3
        if avg_fps < self.DRIFT_FPS_RETRAIN:
            drift_score += 0.4
        drift_score = min(1.0, drift_score)

        if drift_score >= 0.7:
            recommendation = "RETRAIN"
        elif drift_score >= 0.4:
            recommendation = "MONITOR"
        else:
            recommendation = "OK"

        return ModelHealthReport(
            model_name=self.model_name,
            avg_fps=round(avg_fps, 2),
            avg_latency_ms=round(avg_latency, 2),
            total_frames=total_frames,
            uptime_seconds=round(time.time() - self.start_time, 1),
            drift_score=round(drift_score, 3),
            recommendation=recommendation
        )

    def export_to_mlflow(self, run_id: Optional[str] = None):
        """Export metrics hiện tại sang MLflow."""
        try:
            import mlflow
            report = self.get_health_report()
            with mlflow.start_run(run_id=run_id, nested=True):
                mlflow.log_metrics({
                    "avg_fps":         report.avg_fps,
                    "avg_latency_ms":  report.avg_latency_ms,
                    "total_frames":    report.total_frames,
                    "drift_score":     report.drift_score,
                })
        except Exception:
            pass  # MLflow không bắt buộc
