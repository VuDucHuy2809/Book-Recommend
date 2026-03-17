"""
models/shopping_time.py
=======================
MODEL – Theo dõi và tính toán thời gian mua sắm của từng khách hàng.

Mỗi khách hàng được định danh bởi customer_id (ByteTrack track_id).
Lưu timestamp khi xuất hiện lần đầu và cập nhật khi khuất.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CustomerSession:
    """Phiên mua sắm của một khách hàng."""
    customer_id: int
    cam_id: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float  = field(default_factory=time.time)
    is_active: bool = True

    @property
    def duration_seconds(self) -> float:
        """Tổng thời gian hiện diện trong cửa hàng (giây)."""
        return self.last_seen - self.first_seen

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0

    def update(self, timestamp: float = None):
        self.last_seen = timestamp or time.time()

    def close(self):
        self.is_active = False

    def to_dict(self) -> Dict:
        return {
            "customer_id":      self.customer_id,
            "cam_id":           self.cam_id,
            "first_seen":       self.first_seen,
            "last_seen":        self.last_seen,
            "duration_seconds": round(self.duration_seconds, 2),
            "duration_minutes": round(self.duration_minutes, 2),
            "is_active":        self.is_active,
        }


class ShoppingTimeModel:
    """
    Quản lý các phiên mua sắm.

    - Tự động mở phiên khi phát hiện customer_id mới.
    - Cập nhật last_seen mỗi frame.
    - Đóng phiên khi customer_id không xuất hiện quá TIMEOUT giây.
    """

    SESSION_TIMEOUT = 30  # Giây: nếu khách không thấy quá 30s → coi là rời đi

    def __init__(self):
        self._sessions: Dict[int, CustomerSession] = {}   # customer_id → session
        self._closed: List[CustomerSession] = []           # Phiên đã kết thúc

    def update(self, customer_ids: List[int], cam_id: str) -> None:
        """
        Cập nhật trạng thái tất cả khách hàng hiện tại trong frame.

        Args:
            customer_ids: Danh sách customer_id phát hiện trong frame hiện tại.
            cam_id: ID camera.
        """
        now = time.time()

        # Mở phiên mới cho customer_id chưa biết
        for cid in customer_ids:
            if cid not in self._sessions:
                self._sessions[cid] = CustomerSession(
                    customer_id=cid, cam_id=cam_id, first_seen=now, last_seen=now
                )
            else:
                self._sessions[cid].update(now)

        # Đóng phiên cho khách đã rời đi
        timed_out = [
            cid for cid, sess in self._sessions.items()
            if (now - sess.last_seen) > self.SESSION_TIMEOUT
        ]
        for cid in timed_out:
            self._sessions[cid].close()
            self._closed.append(self._sessions.pop(cid))

    def get_active_sessions(self) -> List[CustomerSession]:
        return list(self._sessions.values())

    def get_closed_sessions(self) -> List[CustomerSession]:
        return list(self._closed)

    def get_session(self, customer_id: int) -> Optional[CustomerSession]:
        return self._sessions.get(customer_id)

    def get_stats(self) -> Dict:
        """Thống kê tổng hợp thời gian mua sắm."""
        all_durations = (
            [s.duration_seconds for s in self._sessions.values()] +
            [s.duration_seconds for s in self._closed]
        )
        if not all_durations:
            return {"count": 0, "avg_seconds": 0, "max_seconds": 0}

        return {
            "active_count":    len(self._sessions),
            "total_visited":   len(all_durations),
            "avg_seconds":     round(sum(all_durations) / len(all_durations), 1),
            "max_seconds":     round(max(all_durations), 1),
            "min_seconds":     round(min(all_durations), 1),
        }
