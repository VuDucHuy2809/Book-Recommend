"""
models/behavior_analysis.py
===========================
MODEL – Phân tích hành vi khách hàng trong từng khu vực ngành hàng.

Phát hiện các hành vi:
- Đứng xem (standing/browsing)
- Cầm/đặt sản phẩm (pick up / put down)
- Di chuyển/đi qua (walking)
- Ngồi (sitting)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional

import numpy as np


class BehaviorType(str, Enum):
    BROWSING  = "browsing"    # Đứng xem sản phẩm
    WALKING   = "walking"     # Di chuyển trong cửa hàng
    PICKING   = "picking"     # Cầm sản phẩm lên
    SITTING   = "sitting"     # Đang ngồi
    WAITING   = "waiting"     # Đứng chờ (hàng, thu ngân)
    UNKNOWN   = "unknown"


@dataclass
class BehaviorResult:
    """Kết quả phân tích hành vi của 1 khách hàng."""
    customer_id: int
    behavior: BehaviorType
    confidence: float
    zone_id: Optional[str]
    details: Dict = None

    def to_dict(self) -> Dict:
        return {
            "customer_id": self.customer_id,
            "behavior":    self.behavior.value,
            "confidence":  round(self.confidence, 3),
            "zone_id":     self.zone_id,
            "details":     self.details or {},
        }


class BehaviorAnalysisModel:
    """
    Phân tích hành vi dựa trên tốc độ di chuyển và lịch sử vị trí.

    Phương pháp:
    - Theo dõi centroid qua nhiều frame
    - Tính velocity (tốc độ di chuyển) → phân loại walking/standing
    - Theo dõi thay đổi bbox size → phát hiện cúi xuống, cầm sản phẩm
    """

    WALKING_SPEED_THRESHOLD   = 15.0   # pixel/frame
    BROWSING_SPEED_THRESHOLD  = 5.0
    HISTORY_FRAMES            = 15     # Số frame lưu lịch sử

    def __init__(self):
        # customer_id → deque position history
        self._position_history: Dict[int, List[tuple]] = {}
        self._bbox_history: Dict[int, List[tuple]] = {}

    def analyze(self, detections: List, zone_map: Dict[int, str] = None) -> List[BehaviorResult]:
        """
        Phân tích hành vi tất cả khách hàng.

        Args:
            detections: Danh sách CustomerData.
            zone_map: Dict customer_id → zone_id hiện tại.

        Returns:
            Danh sách BehaviorResult.
        """
        results = []
        zone_map = zone_map or {}

        for customer in detections:
            cid = customer.customer_id
            cx, cy = customer.centroid
            bbox = customer.bbox

            # Cập nhật lịch sử
            if cid not in self._position_history:
                self._position_history[cid] = []
                self._bbox_history[cid] = []

            history = self._position_history[cid]
            history.append((cx, cy))
            if len(history) > self.HISTORY_FRAMES:
                history.pop(0)

            bbox_hist = self._bbox_history[cid]
            bbox_hist.append(bbox)
            if len(bbox_hist) > self.HISTORY_FRAMES:
                bbox_hist.pop(0)

            behavior, confidence = self._classify_behavior(history, bbox_hist)

            results.append(BehaviorResult(
                customer_id=cid,
                behavior=behavior,
                confidence=confidence,
                zone_id=zone_map.get(cid),
            ))

        return results

    def _classify_behavior(
        self, positions: List[tuple], bboxes: List[tuple]
    ) -> tuple:
        """Phân loại hành vi từ lịch sử vị trí."""
        if len(positions) < 3:
            return BehaviorType.UNKNOWN, 0.5

        # Tính vận tốc trung bình
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            velocities.append(np.sqrt(dx**2 + dy**2))

        avg_speed = np.mean(velocities) if velocities else 0

        if avg_speed > self.WALKING_SPEED_THRESHOLD:
            return BehaviorType.WALKING, min(1.0, avg_speed / 30.0)

        if avg_speed < self.BROWSING_SPEED_THRESHOLD:
            # Kiểm tra thay đổi bbox → phát hiện cầm sản phẩm
            if self._detect_picking(bboxes):
                return BehaviorType.PICKING, 0.7
            return BehaviorType.BROWSING, 0.85

        return BehaviorType.WAITING, 0.6

    @staticmethod
    def _detect_picking(bboxes: List[tuple]) -> bool:
        """Phát hiện cầm sản phẩm dựa trên thay đổi diện tích bbox."""
        if len(bboxes) < 5:
            return False
        areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in bboxes]
        area_std = np.std(areas)
        return area_std > np.mean(areas) * 0.08  # 8% biến động

    def get_zone_behavior_summary(
        self, results: List[BehaviorResult]
    ) -> Dict[str, Dict]:
        """Tổng hợp hành vi theo từng zone."""
        summary: Dict[str, Dict] = {}
        for r in results:
            zone = r.zone_id or "unknown"
            if zone not in summary:
                summary[zone] = {}
            b = r.behavior.value
            summary[zone][b] = summary[zone].get(b, 0) + 1
        return summary
