"""
views/shopping_time_view.py
=============================
VIEW – Hiển thị thống kê thời gian mua sắm lên frame.
"""

import cv2
import numpy as np
from typing import Dict


class ShoppingTimeView:
    """Render thống kê thời gian mua sắm."""

    def render(self, frame: np.ndarray, stats: Dict):
        h, w = frame.shape[:2]
        avg_s = stats.get("avg_seconds", 0)
        total = stats.get("total_visited", 0)
        lines = [
            f"Luot tham quan: {total}",
            f"Thoi gian TB : {avg_s//60:.0f}p {avg_s%60:.0f}s",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, h - 50 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
