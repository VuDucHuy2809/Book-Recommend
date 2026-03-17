"""
models/zone_interaction.py
==========================
MODEL – Định nghĩa các khu vực (zone) và theo dõi thời gian tương tác
của từng khách hàng tại mỗi khu vực.

Sử dụng polygon để định nghĩa zone bất kỳ hình dạng nào.
Heatmap được tích lũy từ tọa độ centroid của khách hàng.
"""

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class Zone:
    """Khu vực trong nhà sách."""
    zone_id: str
    name: str
    polygon: np.ndarray  # Shape (N, 2) – tọa độ các đỉnh

    def contains(self, point: Tuple[int, int]) -> bool:
        """Kiểm tra điểm có nằm trong zone không."""
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0


@dataclass
class ZoneDwell:
    """Thời gian lưu trú của 1 khách trong 1 zone."""
    customer_id: int
    zone_id: str
    enter_time: float = field(default_factory=time.time)
    exit_time: Optional[float] = None

    @property
    def dwell_seconds(self) -> float:
        end = self.exit_time or time.time()
        return end - self.enter_time


class ZoneInteractionModel:
    """
    Theo dõi tương tác khách hàng với các zone.

    Tích lũy:
    - Dwell time (thời gian lưu trú) theo customer_id + zone_id
    - Heatmap ma trận để visualize mật độ khách
    """

    def __init__(self, zone_definitions: List[Dict], frame_shape: Tuple = (720, 1280)):
        self.zones: Dict[str, Zone] = {}
        self._load_zones(zone_definitions)

        # Heatmap tích lũy
        h, w = frame_shape[:2]
        self._heatmap = np.zeros((h, w), dtype=np.float32)

        # customer_id → zone_id → ZoneDwell
        self._active_dwells: Dict[int, Dict[str, ZoneDwell]] = {}

        # Tổng dwell time lịch sử theo zone
        self._zone_total_dwell: Dict[str, float] = {zid: 0.0 for zid in self.zones}
        self._zone_visitor_count: Dict[str, int] = {zid: 0 for zid in self.zones}

    def _load_zones(self, zone_definitions: List[Dict]):
        for zdef in zone_definitions:
            polygon = np.array(zdef["polygon"], dtype=np.int32)
            self.zones[zdef["id"]] = Zone(
                zone_id=zdef["id"],
                name=zdef["name"],
                polygon=polygon
            )

    def update(self, detections: List) -> Dict[str, List[int]]:
        """
        Cập nhật vị trí khách, heatmap và dwell time.

        Args:
            detections: Danh sách CustomerData từ CustomerDetectionModel.

        Returns:
            Dict zone_id → [customer_id] đang ở trong zone.
        """
        now = time.time()
        zone_occupancy: Dict[str, List[int]] = {zid: [] for zid in self.zones}
        current_ids = set()

        for customer in detections:
            cid = customer.customer_id
            cx, cy = customer.centroid
            current_ids.add(cid)

            # Cập nhật heatmap
            if 0 <= cy < self._heatmap.shape[0] and 0 <= cx < self._heatmap.shape[1]:
                self._heatmap[cy, cx] += 1

            # Xác định customer đang ở zone nào
            if cid not in self._active_dwells:
                self._active_dwells[cid] = {}

            for zone_id, zone in self.zones.items():
                if zone.contains((cx, cy)):
                    zone_occupancy[zone_id].append(cid)
                    if zone_id not in self._active_dwells[cid]:
                        # Bắt đầu dwell mới
                        self._active_dwells[cid][zone_id] = ZoneDwell(
                            customer_id=cid, zone_id=zone_id, enter_time=now
                        )
                        self._zone_visitor_count[zone_id] += 1
                else:
                    # Khách rời zone → đóng dwell
                    if zone_id in self._active_dwells[cid]:
                        dwell = self._active_dwells[cid].pop(zone_id)
                        dwell.exit_time = now
                        self._zone_total_dwell[zone_id] += dwell.dwell_seconds

        # Dọn dẹp customer đã rời đi
        left_customers = set(self._active_dwells.keys()) - current_ids
        for cid in left_customers:
            for zone_id, dwell in self._active_dwells[cid].items():
                dwell.exit_time = now
                self._zone_total_dwell[zone_id] += dwell.dwell_seconds
            del self._active_dwells[cid]

        return zone_occupancy

    def get_heatmap_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Tạo heatmap overlay lên frame."""
        heatmap_norm = cv2.normalize(
            self._heatmap, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        # Làm mờ để smooth
        heatmap_blur = cv2.GaussianBlur(heatmap_color, (51, 51), 0)
        return cv2.addWeighted(frame, 0.6, heatmap_blur, 0.4, 0)

    def get_zone_stats(self) -> List[Dict]:
        """Thống kê tương tác theo zone."""
        stats = []
        for zone_id, zone in self.zones.items():
            total = self._zone_total_dwell[zone_id]
            visitors = self._zone_visitor_count[zone_id]
            stats.append({
                "zone_id":          zone_id,
                "zone_name":        zone.name,
                "total_dwell_sec":  round(total, 1),
                "visitor_count":    visitors,
                "avg_dwell_sec":    round(total / visitors, 1) if visitors else 0,
            })
        return stats
