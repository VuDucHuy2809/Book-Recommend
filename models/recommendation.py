"""
models/recommendation.py
========================
MODEL – Gợi ý sản phẩm cho TMĐT dựa trên:
1. Hành vi xem sản phẩm (in-store observation)
2. Lịch sử tương tác theo zone → sở thích ngành hàng
3. Demographic (tuổi, giới tính) để cá nhân hóa
4. Collaborative Filtering + Content-based cho TMĐT
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Product:
    """Thông tin sản phẩm sách."""
    product_id: str
    title: str
    category: str
    author: str
    price: float
    tags: List[str]
    zone_id: str
    avg_rating: float = 4.0
    view_count: int = 0
    purchase_count: int = 0
    embedding: Optional[np.ndarray] = None


@dataclass
class RecommendationResult:
    """Kết quả gợi ý sản phẩm."""
    product_id: str
    title: str
    score: float
    reason: str

    def to_dict(self) -> Dict:
        return {
            "product_id": self.product_id,
            "title":      self.title,
            "score":      round(self.score, 3),
            "reason":     self.reason,
        }


class RecommendationModel:
    """
    Hệ thống gợi ý sản phẩm kết hợp In-store behavior + TMĐT.

    Chiến lược:
    - Gợi ý dựa trên zone hiện tại (khách đang ở khu sách thiếu nhi → gợi ý sách thiếu nhi)
    - Gợi ý dựa trên demographic (tuổi, giới tính)
    - Gợi ý liên quan (content-based: cùng tác giả, cùng thể loại)
    - Collaborative filtering từ lịch sử TMĐT (cosine similarity)
    """

    def __init__(self):
        self._product_catalog: Dict[str, Product] = {}
        self._user_behavior: Dict[str, Dict] = {}   # customer_profile_key → behavior
        self._zone_category_map: Dict[str, str] = {
            "z_children": "Sách thiếu nhi",
            "z_novel":    "Tiểu thuyết",
            "z_cashier":  "Bestseller",
        }

    def load_catalog(self, products: List[Product]):
        """Nạp danh mục sản phẩm."""
        for p in products:
            self._product_catalog[p.product_id] = p

    def recommend_by_zone(
        self, zone_id: str, top_n: int = 5
    ) -> List[RecommendationResult]:
        """Gợi ý sản phẩm theo khu vực khách đang đứng."""
        category = self._zone_category_map.get(zone_id)
        if not category:
            return self._get_bestsellers(top_n)

        candidates = [
            p for p in self._product_catalog.values()
            if p.category == category
        ]
        candidates.sort(key=lambda p: p.purchase_count, reverse=True)

        return [
            RecommendationResult(
                product_id=p.product_id,
                title=p.title,
                score=min(1.0, p.purchase_count / 100.0 + p.avg_rating / 10.0),
                reason=f"Phổ biến tại khu {category}"
            )
            for p in candidates[:top_n]
        ]

    def recommend_by_demographic(
        self, age: Optional[int], gender: Optional[str], top_n: int = 5
    ) -> List[RecommendationResult]:
        """Gợi ý dựa trên tuổi và giới tính."""
        if age is None:
            return self._get_bestsellers(top_n)

        preferred_categories = []
        if age < 13:
            preferred_categories = ["Sách thiếu nhi", "Truyện tranh"]
        elif age < 18:
            preferred_categories = ["Sách học sinh", "Kỹ năng sống", "Truyện tranh"]
        elif age < 30:
            preferred_categories = ["Tiểu thuyết", "Văn học nước ngoài", "Self-help"]
        else:
            preferred_categories = ["Kinh tế", "Kỹ năng sống", "Văn học"]

        candidates = [
            p for p in self._product_catalog.values()
            if p.category in preferred_categories
        ]
        candidates.sort(key=lambda p: p.avg_rating * p.purchase_count, reverse=True)

        return [
            RecommendationResult(
                product_id=p.product_id,
                title=p.title,
                score=p.avg_rating / 5.0,
                reason=f"Phù hợp với {'trẻ em' if age < 13 else 'bạn'}"
            )
            for p in candidates[:top_n]
        ]

    def recommend_related(
        self, product_id: str, top_n: int = 5
    ) -> List[RecommendationResult]:
        """Gợi ý sản phẩm liên quan khi khách đang xem 1 sản phẩm."""
        source = self._product_catalog.get(product_id)
        if not source:
            return []

        candidates = [
            p for pid, p in self._product_catalog.items()
            if pid != product_id and (
                p.category == source.category or
                p.author == source.author or
                bool(set(p.tags) & set(source.tags))
            )
        ]

        def relevance(p: Product) -> float:
            score = 0.0
            if p.category == source.category: score += 0.5
            if p.author == source.author:     score += 0.4
            tag_overlap = len(set(p.tags) & set(source.tags))
            score += min(0.3, tag_overlap * 0.1)
            return score

        candidates.sort(key=relevance, reverse=True)
        return [
            RecommendationResult(
                product_id=p.product_id,
                title=p.title,
                score=relevance(p),
                reason=f"Cùng tác giả/thể loại với '{source.title}'"
            )
            for p in candidates[:top_n]
        ]

    def _get_bestsellers(self, top_n: int) -> List[RecommendationResult]:
        """Fallback: trả về bestsellers."""
        candidates = sorted(
            self._product_catalog.values(),
            key=lambda p: p.purchase_count, reverse=True
        )
        return [
            RecommendationResult(
                product_id=p.product_id,
                title=p.title,
                score=min(1.0, p.purchase_count / 200.0),
                reason="Sản phẩm bán chạy"
            )
            for p in candidates[:top_n]
        ]
