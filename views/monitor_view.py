"""
views/monitor_view.py
=======================
VIEW – Dashboard hiệu suất mô hình AI trong frame.
"""

import cv2
import numpy as np

from models.model_monitor import ModelHealthReport


class MonitorView:
    """Render metrics mô hình AI vào góc dưới trái frame."""

    COLORS = {"OK": (0, 200, 0), "MONITOR": (0, 165, 255), "RETRAIN": (0, 0, 255)}

    def render(self, frame: np.ndarray, report: ModelHealthReport):
        h = frame.shape[0]
        color = self.COLORS.get(report.recommendation, (128, 128, 128))
        lines = [
            f"Model : {report.model_name}",
            f"FPS   : {report.avg_fps:.1f}",
            f"Latency: {report.avg_latency_ms:.0f}ms",
            f"Status: {report.recommendation}",
        ]
        for i, line in enumerate(lines):
            y = h - 90 + i * 20
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        color if "Status" in line else (180, 180, 180), 1, cv2.LINE_AA)
