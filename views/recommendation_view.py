"""
views/recommendation_view.py
==============================
VIEW – Hiển thị gợi ý sản phẩm lên màn hình (digital signage / log).
"""

import cv2
import numpy as np
from typing import List

from models.recommendation import RecommendationResult


class RecommendationView:
    """Render danh sách sản phẩm gợi ý lên overlay frame."""

    def render(self, frame: np.ndarray, recommendations: List[RecommendationResult],
               title: str = "Gợi ý cho bạn"):
        if not recommendations:
            return
        h, w = frame.shape[:2]
        panel_x = w - 320
        panel_h = min(200, 30 + len(recommendations) * 30)

        # Panel nền
        cv2.rectangle(frame, (panel_x, h - panel_h - 10),
                      (w - 5, h - 10), (30, 30, 60), -1)
        cv2.rectangle(frame, (panel_x, h - panel_h - 10),
                      (w - 5, h - 10), (100, 100, 200), 1)

        cv2.putText(frame, title, (panel_x + 5, h - panel_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1)

        for i, rec in enumerate(recommendations[:5]):
            y = h - panel_h + 35 + i * 26
            score_bar = int(rec.score * 60)
            cv2.rectangle(frame, (panel_x + 5, y - 10),
                          (panel_x + 5 + score_bar, y - 3), (0, 200, 100), -1)
            title_short = rec.title[:28] + "…" if len(rec.title) > 28 else rec.title
            cv2.putText(frame, f"{i+1}. {title_short}", (panel_x + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
