"""
views/queue_view.py
=====================
VIEW – Hiển thị trạng thái hàng chờ và cảnh báo lên frame.
"""

import cv2
import numpy as np
from typing import Dict


class QueueView:
    """Render trạng thái hàng chờ với indicator màu."""

    def render(self, frame: np.ndarray, status: Dict):
        h, w = frame.shape[:2]
        count = status.get("current_queue_length", 0)
        avg_w = status.get("avg_wait_seconds", 0)

        # Màu theo mức độ
        if count == 0:
            color = (0, 200, 0)     # Xanh = không ai
        elif count <= 3:
            color = (0, 165, 255)   # Cam = bình thường
        else:
            color = (0, 0, 255)     # Đỏ = đông

        # Panel hàng chờ
        panel_x = w - 260
        cv2.rectangle(frame, (panel_x, 10), (w - 10, 100), (40, 40, 40), -1)
        cv2.rectangle(frame, (panel_x, 10), (w - 10, 100), color, 2)
        cv2.putText(frame, "HANG CHO", (panel_x + 10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{count} nguoi cho", (panel_x + 10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, f"Wait avg: {avg_w:.0f}s", (panel_x + 10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Cảnh báo đỏ nếu quá đông
        if count > 5:
            cv2.putText(frame, "!! HANG CHO QUA DAI !!", (w // 2 - 180, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
