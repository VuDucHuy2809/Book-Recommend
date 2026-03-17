"""
models/anomaly_detection.py
===========================
MODEL – Phát hiện hành vi bất thường: chống trộm, gian lận, loitering.

Các hành vi khả nghi được phát hiện:
- Loitering: đứng quá lâu ở một khu vực không phải hàng chờ
- Concealment: hành động che giấu (bbox thay đổi bất thường)
- Tailgating: nhiều người đi qua cửa cùng lúc
- Unattended bag: phát hiện vật bị bỏ lại
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class AnomalyType(str, Enum):
    LOITERING      = "loitering"      # Đứng/đi lại quá lâu ở khu vực khả nghi
    CONCEALMENT    = "concealment"    # Che giấu / cúi người bất thường
    TAILGATING     = "tailgating"     # Nhiều người vào cùng lúc qua 1 lối
    CROWD_DENSITY  = "crowd_density"  # Tụ tập đông bất thường
    UNKNOWN        = "unknown"


@dataclass
class AnomalyEvent:
    """Một sự kiện bất thường được phát hiện."""
    event_id: str
    anomaly_type: AnomalyType
    customer_ids: List[int]
    cam_id: str
    bbox: Optional[Tuple[int, int, int, int]]
    confidence: float
    description: str
    timestamp: float = field(default_factory=time.time)
    is_resolved: bool = False

    def to_dict(self) -> Dict:
        return {
            "event_id":     self.event_id,
            "type":         self.anomaly_type.value,
            "customer_ids": self.customer_ids,
            "cam_id":       self.cam_id,
            "confidence":   round(self.confidence, 3),
            "description":  self.description,
            "timestamp":    self.timestamp,
            "is_resolved":  self.is_resolved,
        }


class AnomalyDetectionModel:
    """
    Phát hiện hành vi bất thường dựa trên rules + thống kê.

    Chiến lược:
    1. Loitering: customer_id đứng > LOITERING_THRESHOLD giây ở zone nhạy cảm
    2. Concealment: bbox height giảm đột ngột (cúi người che giấu đồ)
    3. Crowd density: số người trong zone vượt MAX_DENSITY
    """

    LOITERING_THRESHOLD    = 120     # Giây: đứng > 2 phút tại zone nhạy cảm
    CONCEALMENT_HEIGHT_DROP = 0.35   # 35% giảm chiều cao bbox
    MAX_CROWD_DENSITY      = 8       # Người/zone tối đa trước khi alert
    SENSITIVE_ZONES        = {"z_entrance", "z_cashier"}  # Zone cần giám sát chặt

    def __init__(self):
        self._dwell_start: Dict[int, Dict[str, float]] = {}   # cid → zone_id → enter_time
        self._bbox_history: Dict[int, List[Tuple]] = {}        # cid → bbox history
        self._event_counter = 0

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"EVT_{int(time.time())}_{self._event_counter:04d}"

    def detect(
        self,
        detections: List,
        zone_occupancy: Dict[str, List[int]],
        cam_id: str
    ) -> List[AnomalyEvent]:
        """
        Chạy toàn bộ pipeline phát hiện bất thường.

        Args:
            detections: Danh sách CustomerData.
            zone_occupancy: Dict zone_id → [customer_id].
            cam_id: ID camera.

        Returns:
            Danh sách AnomalyEvent phát hiện được.
        """
        events: List[AnomalyEvent] = []
        now = time.time()

        # ── Phát hiện Loitering ──────────────────────────────────────────
        for zone_id in self.SENSITIVE_ZONES:
            for cid in zone_occupancy.get(zone_id, []):
                if cid not in self._dwell_start:
                    self._dwell_start[cid] = {}
                if zone_id not in self._dwell_start[cid]:
                    self._dwell_start[cid][zone_id] = now
                else:
                    dwell = now - self._dwell_start[cid][zone_id]
                    if dwell > self.LOITERING_THRESHOLD:
                        events.append(AnomalyEvent(
                            event_id=self._next_event_id(),
                            anomaly_type=AnomalyType.LOITERING,
                            customer_ids=[cid],
                            cam_id=cam_id,
                            bbox=None,
                            confidence=min(0.99, 0.6 + dwell / 600),
                            description=f"Khách #{cid} đứng tại {zone_id} "
                                        f"{dwell/60:.1f} phút."
                        ))

        # ── Phát hiện Concealment ────────────────────────────────────────
        for customer in detections:
            cid = customer.customer_id
            bbox = customer.bbox
            if cid not in self._bbox_history:
                self._bbox_history[cid] = []
            hist = self._bbox_history[cid]
            hist.append(bbox)
            if len(hist) > 20:
                hist.pop(0)

            if len(hist) >= 10:
                heights = [(b[3] - b[1]) for b in hist]
                avg_h = np.mean(heights[:-3])
                cur_h = np.mean(heights[-3:])
                if avg_h > 0 and (avg_h - cur_h) / avg_h > self.CONCEALMENT_HEIGHT_DROP:
                    events.append(AnomalyEvent(
                        event_id=self._next_event_id(),
                        anomaly_type=AnomalyType.CONCEALMENT,
                        customer_ids=[cid],
                        cam_id=cam_id,
                        bbox=tuple(bbox),
                        confidence=0.70,
                        description=f"Khách #{cid} có hành vi cúi người bất thường."
                    ))

        # ── Phát hiện Crowd Density ──────────────────────────────────────
        for zone_id, occupants in zone_occupancy.items():
            if len(occupants) >= self.MAX_CROWD_DENSITY:
                events.append(AnomalyEvent(
                    event_id=self._next_event_id(),
                    anomaly_type=AnomalyType.CROWD_DENSITY,
                    customer_ids=occupants,
                    cam_id=cam_id,
                    bbox=None,
                    confidence=0.90,
                    description=f"Tụ tập {len(occupants)} người tại zone {zone_id}."
                ))

        return events
