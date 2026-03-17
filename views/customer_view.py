"""
views/customer_view.py
========================
VIEW – Render thông tin khách hàng lên frame camera.
Vẽ bounding box, Customer ID, tuổi, giới tính và counter.
"""

import cv2
import numpy as np
from typing import Dict, List

from models.customer_detection import CustomerData


class CustomerView:
    """Render kết quả nhận diện khách hàng lên frame."""

    # Màu theo giới tính
    COLOR_MALE    = (255, 150, 50)   # BGR: cam nhạt
    COLOR_FEMALE  = (200, 100, 255)  # BGR: tím nhạt
    COLOR_UNKNOWN = (180, 180, 180)  # BGR: xám

    GENDER_ICON = {"male": "♂", "female": "♀"}
    AGE_GROUP_COLOR = {
        "child":  (0, 200, 255),
        "teen":   (0, 255, 100),
        "adult":  (255, 200, 0),
        "senior": (100, 100, 255),
    }

    def render(self, frame: np.ndarray, detections: List[CustomerData], stats: Dict):
        """Vẽ tất cả thông tin khách hàng lên frame."""
        for customer in detections:
            self._draw_customer(frame, customer)
        self._draw_counter(frame, stats)

    def _draw_customer(self, frame: np.ndarray, c: CustomerData):
        x1, y1, x2, y2 = c.bbox

        # Chọn màu theo giới tính
        color = (
            self.COLOR_MALE if c.gender == "male"
            else self.COLOR_FEMALE if c.gender == "female"
            else self.COLOR_UNKNOWN
        )

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label: ID + giới tính + tuổi
        gender_sym = self.GENDER_ICON.get(c.gender or "", "?")
        age_str = f"{c.age}y" if c.age else "?y"
        label = f"#{c.customer_id} {gender_sym} {age_str}"

        # Nền label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Dấu chấm centroid
        cv2.circle(frame, c.centroid, 4, color, -1)

    @staticmethod
    def _draw_counter(frame: np.ndarray, stats: Dict):
        """Vẽ bảng thống kê góc trên trái."""
        lines = [
            f"Hien tai : {stats.get('current', 0)} nguoi",
            f"Tong luot: {stats.get('total_unique', 0)} nguoi",
            f"Dinh diem: {stats.get('peak', 0)} nguoi",
        ]
        y_start = 30
        for i, line in enumerate(lines):
            y = y_start + i * 24
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
