"""
views/zone_view.py
====================
VIEW – Vẽ zone boundaries, nhãn zone và occupancy lên frame.
"""

import cv2
import numpy as np
from typing import Dict, List


class ZoneView:
    """Render zone polygons và occupancy lên frame."""

    ZONE_COLORS = [
        (100, 255, 100),   # xanh lá
        (100, 200, 255),   # xanh lam
        (255, 200, 100),   # vàng
        (200, 100, 255),   # tím
        (100, 255, 200),   # cyan
    ]
    ALPHA = 0.25  # Độ trong suốt fill

    def __init__(self, zone_definitions: List[Dict]):
        self.zone_definitions = zone_definitions

    def render(self, frame: np.ndarray, zone_occupancy: Dict[str, List[int]], zones: Dict):
        overlay = frame.copy()
        for i, (zone_id, zone) in enumerate(zones.items()):
            color = self.ZONE_COLORS[i % len(self.ZONE_COLORS)]
            count = len(zone_occupancy.get(zone_id, []))

            # Fill polygon
            cv2.fillPoly(overlay, [zone.polygon], color)

            # Contour
            cv2.polylines(frame, [zone.polygon], True, color, 2)

            # Label tại tâm
            M = cv2.moments(zone.polygon)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(frame, f"{zone.name}", (cx - 40, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"{count} nguoi", (cx - 30, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Blend overlay
        cv2.addWeighted(overlay, self.ALPHA, frame, 1 - self.ALPHA, 0, frame)
