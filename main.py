"""
Fahasa AI Camera System
=======================
Entry point – khởi tạo và điều phối toàn bộ hệ thống multi-camera.

Hỗ trợ kết nối đồng thời nhiều camera (RTSP / USB / file video).
Mỗi camera chạy trong một thread riêng biệt.
"""

import sys
import signal
import threading
import time
from typing import List, Dict, Any

from loguru import logger

from config.settings import Settings
from config.logging_config import setup_logging
from config.database import init_database
from services.camera_service import CameraService
from services.analytics_service import AnalyticsService
from services.alert_service import AlertService

from controllers.customer_controller import CustomerController
from controllers.shopping_time_controller import ShoppingTimeController
from controllers.zone_controller import ZoneController
from controllers.behavior_controller import BehaviorController
from controllers.queue_controller import QueueController
from controllers.anomaly_controller import AnomalyController
from controllers.recommendation_controller import RecommendationController
from controllers.monitor_controller import MonitorController


class FahasaAICamera:
    """
    Orchestrator chính cho hệ thống Camera AI Fahasa.
    Quản lý vòng đời của mọi camera và module xử lý.
    """

    def __init__(self):
        self.settings = Settings()
        setup_logging(self.settings.LOG_LEVEL)
        init_database(self.settings.DATABASE_URL)

        self.running = False
        self.camera_threads: List[threading.Thread] = []
        self.camera_services: List[CameraService] = []

        # Shared services
        self.alert_service = AlertService(self.settings)
        self.analytics_service = AnalyticsService(self.settings)

        # Controllers (khởi tạo một lần, dùng chung cho tất cả cameras)
        self.controllers = self._init_controllers()

        logger.info(f"🚀 Fahasa AI Camera System khởi tạo thành công.")
        logger.info(f"📷 Số camera cấu hình: {len(self.settings.CAMERA_SOURCES)}")

    def _init_controllers(self) -> Dict[str, Any]:
        """Khởi tạo tất cả controllers."""
        return {
            "customer":     CustomerController(self.settings, self.alert_service),
            "shopping_time":ShoppingTimeController(self.settings),
            "zone":         ZoneController(self.settings, self.alert_service),
            "behavior":     BehaviorController(self.settings),
            "queue":        QueueController(self.settings, self.alert_service),
            "anomaly":      AnomalyController(self.settings, self.alert_service),
            "recommendation":RecommendationController(self.settings),
            "monitor":      MonitorController(self.settings, self.alert_service),
        }

    def _process_camera(self, camera_config: Dict[str, Any]):
        """
        Vòng lặp xử lý chính cho một camera đơn lẻ.
        Chạy trong thread riêng.
        """
        cam_id = camera_config["id"]
        cam_name = camera_config["name"]
        cam_source = camera_config["source"]

        logger.info(f"[{cam_id}] 🎥 Khởi động camera: {cam_name} (source={cam_source})")

        cam_service = CameraService(camera_config)
        self.camera_services.append(cam_service)

        if not cam_service.connect():
            logger.error(f"[{cam_id}] ❌ Không thể kết nối camera. Bỏ qua.")
            return

        while self.running:
            ret, frame = cam_service.read_frame()
            if not ret or frame is None:
                logger.warning(f"[{cam_id}] ⚠️ Mất tín hiệu camera, đang thử lại...")
                time.sleep(2)
                cam_service.reconnect()
                continue

            # ── Pipeline xử lý từng frame ────────────────────────────────
            try:
                # 1. Nhận diện khách hàng + gán Customer ID
                detections = self.controllers["customer"].process(frame, cam_id)

                # 2. Tính thời gian mua sắm
                self.controllers["shopping_time"].process(detections, cam_id)

                # 3. Tương tác theo khu vực (heatmap)
                self.controllers["zone"].process(detections, frame, cam_id)

                # 4. Phân tích hành vi
                self.controllers["behavior"].process(detections, frame, cam_id)

                # 5. Quản lý hàng chờ
                self.controllers["queue"].process(detections, frame, cam_id)

                # 6. Phát hiện hành vi bất thường
                self.controllers["anomaly"].process(detections, frame, cam_id)

                # 7. Theo dõi hiệu suất mô hình
                self.controllers["monitor"].log_frame_metrics(cam_id)

            except Exception as e:
                logger.error(f"[{cam_id}] ❌ Lỗi xử lý frame: {e}")

        cam_service.release()
        logger.info(f"[{cam_id}] ✅ Camera dừng lại.")

    def start(self):
        """Khởi động toàn bộ hệ thống với tất cả cameras."""
        self.running = True
        logger.info("=" * 60)
        logger.info("  FAHASA AI CAMERA SYSTEM - KHỞI ĐỘNG")
        logger.info("=" * 60)

        if not self.settings.CAMERA_SOURCES:
            logger.error("❌ Không có camera nào được cấu hình. Kiểm tra CAMERA_SOURCES.")
            sys.exit(1)

        # Tạo thread riêng cho mỗi camera
        for cam_config in self.settings.CAMERA_SOURCES:
            thread = threading.Thread(
                target=self._process_camera,
                args=(cam_config,),
                name=f"Camera-{cam_config['id']}",
                daemon=True
            )
            self.camera_threads.append(thread)
            thread.start()
            logger.info(f"✅ Thread [{cam_config['id']}] đã khởi động.")

        # Bắt đầu analytics service (báo cáo định kỳ)
        self.analytics_service.start_periodic_report()

        logger.info(f"🟢 Hệ thống đang chạy với {len(self.camera_threads)} camera(s).")

    def stop(self):
        """Dừng toàn bộ hệ thống an toàn."""
        logger.info("🛑 Đang dừng hệ thống...")
        self.running = False

        for thread in self.camera_threads:
            thread.join(timeout=5)

        for cam_service in self.camera_services:
            cam_service.release()

        self.analytics_service.stop()
        logger.info("✅ Hệ thống đã dừng hoàn toàn.")

    def wait(self):
        """Block main thread cho đến khi nhận tín hiệu dừng."""
        while self.running:
            time.sleep(1)


def handle_signal(signum, frame, app: FahasaAICamera):
    """Xử lý Ctrl+C / SIGTERM để dừng hệ thống gracefully."""
    logger.info(f"\n🔔 Nhận tín hiệu {signum}. Đang dừng hệ thống...")
    app.stop()
    sys.exit(0)


def main():
    app = FahasaAICamera()

    # Đăng ký signal handlers
    signal.signal(signal.SIGINT,  lambda s, f: handle_signal(s, f, app))
    signal.signal(signal.SIGTERM, lambda s, f: handle_signal(s, f, app))

    app.start()
    app.wait()


if __name__ == "__main__":
    main()
