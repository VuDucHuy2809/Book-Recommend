"""
views/behavior_view.py
========================
VIEW – Render nhãn hành vi lên frame.
"""

import cv2
import numpy as np
from typing import List

from models.behavior_analysis import BehaviorResult, BehaviorType


class BehaviorView:
    BEHAVIOR_COLORS = {
        BehaviorType.BROWSING: (0, 200, 0),
        BehaviorType.WALKING:  (200, 200, 0),
        BehaviorType.PICKING:  (0, 100, 255),
        BehaviorType.SITTING:  (150, 150, 255),
        BehaviorType.WAITING:  (0, 165, 255),
        BehaviorType.UNKNOWN:  (128, 128, 128),
    }

    def render(self, frame: np.ndarray, results: List[BehaviorResult]):
        for r in results:
            color = self.BEHAVIOR_COLORS.get(r.behavior, (128, 128, 128))
            # Chỉ đặt text khi có thông tin vị trí (cần ghép với detection)
            # Render behavior label dưới bbox trong customer_view
