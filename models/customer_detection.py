"""
models/customer_detection.py
============================
MODEL – Nhận diện khách hàng, gán Customer ID duy nhất, phân tích tuổi/giới tính.

Sử dụng:
- YOLOv11 (Ultralytics latest) để detect người
- ByteTrack để tracking và gán Customer ID liên tục
- DeepFace / InsightFace để phân tích tuổi, giới tính
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from utils.tracker_utils import ByteTrackWrapper


@dataclass
class CustomerData:
    """Đại diện cho một khách hàng được phát hiện trong frame."""
    customer_id: int                     # ID duy nhất từ tracker (ByteTrack track_id)
    bbox: Tuple[int, int, int, int]      # (x1, y1, x2, y2)
    confidence: float                    # Độ tin cậy detection YOLO
    centroid: Tuple[int, int]            # Tâm bounding box
    gender: Optional[str] = None         # "male" | "female" | None
    age: Optional[int] = None            # Tuổi ước tính
    age_group: Optional[str] = None      # "child" | "teen" | "adult" | "senior"
    first_seen: float = field(default_factory=time.time)
    last_seen: float  = field(default_factory=time.time)
    cam_id: str = ""

    @property
    def cx(self) -> int:
        return self.centroid[0]

    @property
    def cy(self) -> int:
        return self.centroid[1]

    def to_dict(self) -> Dict:
        return {
            "customer_id": self.customer_id,
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 3),
            "gender": self.gender,
            "age": self.age,
            "age_group": self.age_group,
            "cam_id": self.cam_id,
        }


class CustomerDetectionModel:
    """
    Wrapper tích hợp YOLOv11 + ByteTrack + DeepFace.

    Luồng xử lý:
      Frame → YOLO detect persons → ByteTrack assign Customer ID
            → DeepFace phân tích tuổi/giới tính (async, mỗi N giây/customer)
    """

    YOLO_PERSON_CLASS = 0          # Class ID 'person' trong COCO
    FACE_ANALYSIS_INTERVAL = 30    # Giây giữa 2 lần phân tích face cho cùng 1 customer

    def __init__(self, model_path: str = "yolo11n.pt",
                 confidence: float = 0.5,
                 face_analyzer: str = "deepface"):
        """
        Args:
            model_path: Đường dẫn đến weights YOLOv11 (tự động download nếu chưa có).
            confidence: Ngưỡng confidence tối thiểu.
            face_analyzer: "deepface" hoặc "insightface".
        """
        self.confidence = confidence
        self.face_analyzer_type = face_analyzer
        self._model = None
        self._tracker = ByteTrackWrapper()
        self._face_cache: Dict[int, Dict] = {}    # customer_id → {gender, age, last_analyzed}
        self._model_path = model_path
        self._load_model()

    def _load_model(self):
        """Lazy load YOLOv11 model."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            logger.info(f"✅ YOLOv11 model loaded: {self._model_path}")
        except ImportError:
            logger.warning("⚠️ ultralytics chưa được cài. Chạy: pip install ultralytics")
        except Exception as e:
            logger.error(f"❌ Không tải được YOLO model: {e}")

    def detect(self, frame: np.ndarray, cam_id: str = "") -> List[CustomerData]:
        """
        Phát hiện, tracking và phân tích tất cả khách hàng trong frame.

        Args:
            frame: BGR frame từ camera.
            cam_id: ID camera để ghi log.

        Returns:
            Danh sách CustomerData, mỗi item là 1 khách hàng với Customer ID duy nhất.
        """
        if self._model is None or frame is None:
            return []

        customers: List[CustomerData] = []

        try:
            # ── Bước 1: YOLO detect với ByteTrack tracking ──────────────
            results = self._model.track(
                frame,
                persist=True,           # Giữ track_id xuyên suốt các frame
                tracker="bytetrack.yaml",
                classes=[self.YOLO_PERSON_CLASS],
                conf=self.confidence,
                verbose=False
            )

            if not results or results[0].boxes is None:
                return []

            boxes = results[0].boxes

            for box in boxes:
                # Lấy track_id (Customer ID duy nhất)
                if box.id is None:
                    continue

                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf.item())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # ── Bước 2: Phân tích tuổi/giới tính ───────────────────
                gender, age, age_group = self._get_face_analysis(
                    frame, track_id, (x1, y1, x2, y2)
                )

                customer = CustomerData(
                    customer_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    centroid=(cx, cy),
                    gender=gender,
                    age=age,
                    age_group=age_group,
                    last_seen=time.time(),
                    cam_id=cam_id
                )
                customers.append(customer)

        except Exception as e:
            logger.error(f"[{cam_id}] Lỗi detect customer: {e}")

        return customers

    def _get_face_analysis(
        self,
        frame: np.ndarray,
        track_id: int,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """
        Phân tích tuổi / giới tính từ vùng khuôn mặt.
        Kết quả được cache theo track_id để tránh gọi lại quá thường xuyên.
        """
        now = time.time()
        cached = self._face_cache.get(track_id)

        if cached and (now - cached["last_analyzed"]) < self.FACE_ANALYSIS_INTERVAL:
            return cached["gender"], cached["age"], cached["age_group"]

        try:
            x1, y1, x2, y2 = bbox
            # Mở rộng bbox lên trên để bắt khuôn mặt
            face_y1 = max(0, y1 - 20)
            face_y2 = min(frame.shape[0], y1 + (y2 - y1) // 3)
            face_crop = frame[face_y1:face_y2, x1:x2]

            if face_crop.size == 0:
                return None, None, None

            gender, age = self._analyze_face(face_crop)
            age_group = self._classify_age_group(age)

            self._face_cache[track_id] = {
                "gender": gender,
                "age": age,
                "age_group": age_group,
                "last_analyzed": now
            }
            return gender, age, age_group

        except Exception:
            return None, None, None

    def _analyze_face(
        self, face_img: np.ndarray
    ) -> Tuple[Optional[str], Optional[int]]:
        """Gọi DeepFace hoặc InsightFace để lấy tuổi + giới tính."""
        if self.face_analyzer_type == "deepface":
            return self._analyze_with_deepface(face_img)
        return self._analyze_with_insightface(face_img)

    @staticmethod
    def _analyze_with_deepface(
        face_img: np.ndarray
    ) -> Tuple[Optional[str], Optional[int]]:
        try:
            from deepface import DeepFace
            analysis = DeepFace.analyze(
                face_img,
                actions=["age", "gender"],
                enforce_detection=False,
                silent=True
            )
            if isinstance(analysis, list):
                analysis = analysis[0]
            gender = "male" if analysis["dominant_gender"] == "Man" else "female"
            age = int(analysis["age"])
            return gender, age
        except Exception:
            return None, None

    @staticmethod
    def _analyze_with_insightface(
        face_img: np.ndarray
    ) -> Tuple[Optional[str], Optional[int]]:
        try:
            import insightface
            app = insightface.app.FaceAnalysis()
            app.prepare(ctx_id=0, det_size=(640, 640))
            faces = app.get(face_img)
            if faces:
                face = faces[0]
                gender = "male" if face.gender == 1 else "female"
                age = int(face.age)
                return gender, age
        except Exception:
            pass
        return None, None

    @staticmethod
    def _classify_age_group(age: Optional[int]) -> Optional[str]:
        if age is None:
            return None
        if age < 13:
            return "child"
        if age < 18:
            return "teen"
        if age < 60:
            return "adult"
        return "senior"
