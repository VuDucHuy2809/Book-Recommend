"""
controllers/recommendation_controller.py
==========================================
CONTROLLER – Gợi ý sản phẩm dựa trên hành vi tại cửa hàng và TMĐT.
"""

from typing import Dict, List, Optional

from loguru import logger

from models.recommendation import RecommendationModel, RecommendationResult
from models.customer_detection import CustomerData
from views.recommendation_view import RecommendationView
from config.settings import Settings


class RecommendationController:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = RecommendationModel()
        self._view = RecommendationView()
        logger.info("✅ RecommendationController khởi tạo.")

    def load_product_catalog(self, products: List):
        """Nạp danh mục sản phẩm vào recommendation engine."""
        self._model.load_catalog(products)
        logger.info(f"📚 Loaded {len(products)} sản phẩm vào catalog.")

    def recommend_for_customer(
        self,
        customer: CustomerData,
        current_zone_id: Optional[str] = None,
        viewed_product_id: Optional[str] = None,
        top_n: int = 5
    ) -> List[RecommendationResult]:
        """
        Gợi ý sản phẩm cho 1 khách hàng cụ thể.

        Ưu tiên: viewed_product > zone > demographic
        """
        if viewed_product_id:
            recs = self._model.recommend_related(viewed_product_id, top_n)
        elif current_zone_id:
            recs = self._model.recommend_by_zone(current_zone_id, top_n)
        else:
            recs = self._model.recommend_by_demographic(
                customer.age, customer.gender, top_n
            )

        logger.debug(
            f"🛒 Customer #{customer.customer_id} → {len(recs)} gợi ý "
            f"(zone={current_zone_id}, product={viewed_product_id})"
        )
        return recs

    def recommend_for_ecommerce(
        self,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        zone_id: Optional[str] = None,
        top_n: int = 10
    ) -> List[Dict]:
        """API endpoint cho TMĐT: trả về gợi ý dạng JSON."""
        if zone_id:
            recs = self._model.recommend_by_zone(zone_id, top_n)
        else:
            recs = self._model.recommend_by_demographic(age, gender, top_n)
        return [r.to_dict() for r in recs]
