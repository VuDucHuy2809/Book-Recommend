"""
models/queue_management.py
==========================
MODEL – Phát hiện hàng chờ, đếm số người, đo thời gian chờ.

Khu vực hàng chờ được định nghĩa như Zone.
Model theo dõi:
- Số người trong khu vực quầy thu ngân
- Thời gian chờ trung bình (từ khi vào queue → rời queue)
- Tốc độ phục vụ (throughput / giờ)
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QueueEntry:
    """Một người đang trong hàng chờ."""
    customer_id: int
    queue_zone_id: str
    enter_time: float = field(default_factory=time.time)
    exit_time: Optional[float] = None

    @property
    def wait_seconds(self) -> float:
        end = self.exit_time or time.time()
        return end - self.enter_time


class QueueManagementModel:
    """
    Quản lý hàng chờ tại các quầy thanh toán / dịch vụ.

    Logic:
    - Khi customer_id xuất hiện trong queue_zone → thêm vào hàng chờ
    - Khi rời zone → tính toán wait_time và ghi nhận
    - Theo dõi số lượng người trong hàng theo thời gian thực
    """

    def __init__(self, queue_zone_ids: List[str]):
        """
        Args:
            queue_zone_ids: Danh sách zone_id được coi là khu vực hàng chờ.
        """
        self.queue_zone_ids = set(queue_zone_ids)
        self._active_queue: Dict[int, QueueEntry] = {}  # customer_id → entry
        self._history: List[QueueEntry] = []            # Đã phục vụ xong

    def update(self, zone_occupancy: Dict[str, List[int]]) -> Dict:
        """
        Cập nhật trạng thái hàng chờ từ zone_occupancy.

        Args:
            zone_occupancy: Output từ ZoneInteractionModel.update()

        Returns:
            Dict trạng thái hàng chờ hiện tại.
        """
        now = time.time()
        current_in_queue = set()

        for zone_id in self.queue_zone_ids:
            for cid in zone_occupancy.get(zone_id, []):
                current_in_queue.add(cid)
                if cid not in self._active_queue:
                    self._active_queue[cid] = QueueEntry(
                        customer_id=cid, queue_zone_id=zone_id, enter_time=now
                    )

        # Người đã rời hàng chờ
        left = set(self._active_queue.keys()) - current_in_queue
        for cid in left:
            entry = self._active_queue.pop(cid)
            entry.exit_time = now
            self._history.append(entry)

        return self.get_status()

    def get_status(self) -> Dict:
        """Trạng thái hàng chờ hiện tại."""
        current_count = len(self._active_queue)
        wait_times = [e.wait_seconds for e in self._history[-50:]]  # 50 lần gần nhất

        return {
            "current_queue_length": current_count,
            "avg_wait_seconds":     round(sum(wait_times) / len(wait_times), 1) if wait_times else 0,
            "max_wait_seconds":     round(max(wait_times), 1) if wait_times else 0,
            "total_served":         len(self._history),
            "customers_in_queue":   list(self._active_queue.keys()),
        }

    def is_overcrowded(self, threshold: int = 5) -> bool:
        return len(self._active_queue) >= threshold

    def get_longest_wait(self) -> Optional[QueueEntry]:
        if not self._active_queue:
            return None
        return max(self._active_queue.values(), key=lambda e: e.wait_seconds)
