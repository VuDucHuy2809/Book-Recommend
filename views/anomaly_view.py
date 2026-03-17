"""
views/anomaly_view.py
=======================
VIEW – Vẽ cảnh báo bất thường lên frame (đỏ, highlight).
"""

import cv2
import numpy as np
from typing import List

from models.anomaly_detection import AnomalyEvent, AnomalyType


class AnomalyView:
    """Render anomaly alerts lên frame với highlight màu đỏ."""

    SEVERITY_COLORS = {
        AnomalyType.LOITERING:     (0, 0, 255),     # Đỏ đậm
        AnomalyType.CONCEALMENT:   (0, 50, 255),    # Đỏ cam
        AnomalyType.TAILGATING:    (0, 165, 255),   # Cam
        AnomalyType.CROWD_DENSITY: (0, 0, 200),     # Đỏ
        AnomalyType.UNKNOWN:       (128, 0, 128),   # Tím
    }

    def render(self, frame: np.ndarray, events: List[AnomalyEvent], detections: List):
        """Vẽ highlight đỏ cho khách khả nghi và banner cảnh báo."""
        detection_map = {d.customer_id: d for d in detections}

        for event in events:
            color = self.SEVERITY_COLORS.get(event.anomaly_type, (0, 0, 255))

            # Highlight bbox của customer khả nghi
            for cid in event.customer_ids:
                if cid in detection_map:
                    customer = detection_map[cid]
                    x1, y1, x2, y2 = customer.bbox
                    # Viền đỏ dày
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                    # Icon cảnh báo
                    cv2.putText(frame, f"⚠ {event.anomaly_type.value}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # Banner cảnh báo phía trên frame
            if event.anomaly_type == AnomalyType.CONCEALMENT:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (0, 0, 180), -1)
                cv2.putText(frame, f"CANH BAO AN NINH: {event.description[:60]}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (255, 255, 255), 2, cv2.LINE_AA)
