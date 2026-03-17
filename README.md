# 📷 Fahasa AI Camera System

Hệ thống **Camera AI** cho chuỗi nhà sách **Fahasa** — phân tích hành vi khách hàng, quản lý luồng khách, phát hiện bất thường và gợi ý sản phẩm thông minh.

## 🏗️ Kiến trúc hệ thống (MVC)

```
fahasa-ai-camera/
├── main.py                    # Entry point – multi-camera orchestration
├── config/                    # Cấu hình hệ thống
├── models/                    # M – Business logic, AI models
├── views/                     # V – Rendering output lên frame/API
├── controllers/               # C – Điều phối xử lý
├── services/                  # Shared services (camera, alert, analytics)
├── utils/                     # Tiện ích dùng chung
├── database/                  # ORM models + migrations
└── tests/                     # Unit tests
```

## 🚀 Tính năng chính

| Module | Chức năng |
|---|---|
| 👥 Customer Detection | Nhận diện khách, gán **Customer ID** duy nhất, phân tích tuổi/giới tính |
| ⏱️ Shopping Time | Đo thời gian mua sắm của từng khách hàng |
| 🗺️ Zone Interaction | Heatmap tương tác tại các khu vực/sản phẩm |
| 🧠 Behavior Analysis | Phân tích hành vi theo từng khu vực ngành hàng |
| 🚶 Queue Management | Quản lý xếp hàng và luồng khách |
| 🔴 Anomaly Detection | Phát hiện hành vi bất thường (chống trộm) |
| 🛒 Recommendation | Gợi ý sản phẩm cá nhân hóa cho TMĐT |
| 📊 Model Monitor | MLOps – theo dõi hiệu suất mô hình AI |

## ⚙️ Cài đặt

```bash
# 1. Clone project
git clone <repo-url>
cd fahasa-ai-camera

# 2. Tạo virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Cấu hình environment
cp .env.example .env
# Chỉnh sửa .env với các thông số thực tế

# 5. Khởi tạo database
alembic upgrade head

# 6. Chạy hệ thống
python main.py
```

## 📷 Kết nối nhiều Camera

Chỉnh file `.env` hoặc `config/settings.py`:

```python
CAMERA_SOURCES = [
    {"id": "cam_001", "name": "Cửa vào chính",    "source": "rtsp://..."},
    {"id": "cam_002", "name": "Khu sách thiếu nhi","source": "rtsp://..."},
    {"id": "cam_003", "name": "Quầy thanh toán",   "source": 0},  # USB cam
]
```

## 🧪 Chạy tests

```bash
python -m pytest tests/ -v
```

## 🛠️ Công nghệ sử dụng

- **AI Model**: YOLOv11 (Ultralytics latest), DeepFace, InsightFace
- **Tracking**: ByteTrack (Customer ID gán theo track)
- **Backend**: FastAPI + SQLAlchemy + PostgreSQL + Redis
- **MLOps**: MLflow + Prometheus
- **Language**: Python 3.10+
