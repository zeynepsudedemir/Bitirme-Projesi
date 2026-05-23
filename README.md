

# DroneVision — Aerial Object Detection Platform

<div align="center">

![DroneVision Banner](https://img.shields.io/badge/DroneVision-Aerial%20AI-0a0a0a?style=for-the-badge&logo=drone&logoColor=white)

**Real-time multi-model object detection for drone-captured aerial imagery**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=flat-square&logo=postgresql&logoColor=white)](https://postgresql.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 📖 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Models](#models)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---

## Overview

**DroneVision** is a containerized, microservices-based platform for real-time aerial object detection. Designed for the [VisDrone dataset](https://github.com/VisDrone/VisDrone-Dataset) benchmark classes, it supports dynamic switching between three state-of-the-art deep learning models — **YOLOv5**, **YOLOv8**, and **Faster R-CNN** — all served through a unified REST API and backed by a PostgreSQL detection history database.

https://github.com/user-attachments/assets/fa919b0e-8715-4504-8719-0539002bbc98
### Use Cases
- Drone-based traffic monitoring and vehicle counting
- Crowd density analysis from aerial perspectives
- Perimeter security with multi-class object alerting
- Smart city infrastructure surveillance

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DroneVision Stack                        │
│                                                                 │
│  ┌──────────────┐    ┌────────────────────┐    ┌─────────────┐  │
│  │   Frontend   │    │  Inference Service │    │  Auth       │  │
│  │  (React/Vite)│◄──►│  FastAPI + PyTorch │◄──►│  Service    │  │
│  │  (Planned)   │    │  Port: 8001        │    │  (Planned)  │  │
│  └──────────────┘    └────────┬───────────┘    └─────────────┘  │
│                               │                                 │
│                    ┌──────────▼──────────┐                      │
│                    │   PostgreSQL 16     │                      │
│                    │   drone_net bridge  │                      │
│                    │   Port: 5432        │                      │
│                    └─────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

All services communicate over the isolated `drone_net` Docker bridge network.

---

## Features

| Feature | Status |
|---|---|
| Multi-model inference (YOLOv5 / YOLOv8 / Faster R-CNN) | ✅ Live |
| Synchronous REST inference endpoint | ✅ Live |
| Detection persistence to PostgreSQL | ✅ Live |
| GPU acceleration (NVIDIA CUDA) | ✅ Live |
| Health check endpoint | ✅ Live |
| React dashboard with live visualization | ✅ Live |
| Model confidence / class filtering | ✅ Live  |
| SAHI (Sliced Aided Hyper Inference) | ✅ Live  |
| Explainable AI — Grad-CAM overlays | ✅ Live |
| JWT / OAuth2 Auth Service | 🔧 In Progress |
| Real-time RTSP stream ingestion | 🔧 In Progress |


---

## Models

The platform supports three interchangeable detection models, all trained on the VisDrone benchmark:

| Model | Format | Classes | Notes |
|---|---|---|---|
| **YOLOv5** | `.pt` (Ultralytics) | 10 | Fast inference, good for edge cases |
| **YOLOv8** | `.pt` (Ultralytics) | 10 | Best accuracy/speed tradeoff; default model |
| **Faster R-CNN** | `.pth` (PyTorch) | 10 + background | ResNet-101 + FPN backbone, highest accuracy |

### Detection Classes
```
pedestrian · people · bicycle · car · van
truck · tricycle · awning-tricycle · bus · motor
```

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 24.x
- [Docker Compose](https://docs.docker.com/compose/) ≥ 2.x
- NVIDIA GPU with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- Trained model weights placed under `services/inference-service/app/models/`:
  - `yolov5.pt`
  - `yolov8.pt`
  - `faster_rcnn.pth`

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-org/dronevision.git
cd dronevision
```

### 2. Add your model weights

```bash
mkdir -p services/inference-service/app/models
# Copy your trained weights here:
cp /path/to/yolov5.pt     services/inference-service/app/models/
cp /path/to/yolov8.pt     services/inference-service/app/models/
cp /path/to/faster_rcnn.pth services/inference-service/app/models/
```

### 3. Build and run

```bash
docker compose up --build
```

### 4. Verify the services are healthy

```bash
curl http://localhost:8001/health
```

Expected response:
```json
{
  "status": "ok",
  "gpu_available": true,
  "models_loaded": ["yolov5", "yolov8", "faster_rcnn"]
}
```

---

## API Reference

### `GET /health`
Returns service status, GPU availability, and loaded models.

---

### `POST /api/v1/infer/sync`

Run inference on an uploaded image frame.

**Parameters**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | `UploadFile` | ✅ | — | Image file (JPEG / PNG) |
| `model_name` | `string` | ❌ | `yolov8` | One of `yolov5`, `yolov8`, `faster_rcnn` |

**Example (cURL)**

```bash
curl -X POST http://localhost:8001/api/v1/infer/sync \
  -F "file=@/path/to/drone_frame.jpg" \
  -F "model_name=yolov8"
```

**Example Response**

```json
{
  "frame_id": "a3f2e1b0-...",
  "model_name": "yolov8",
  "detections": [
    {
      "label": "car",
      "confidence": 0.921,
      "bbox": { "x1": 142, "y1": 88, "x2": 198, "y2": 120 }
    },
    {
      "label": "pedestrian",
      "confidence": 0.734,
      "bbox": { "x1": 310, "y1": 201, "x2": 325, "y2": 238 }
    }
  ]
}
```

All detections are automatically persisted to the PostgreSQL `detections` table.

---

## Project Structure

```
dronevision/
├── docker-compose.yml
├── services/
│   └── inference-service/
│       ├── Dockerfile
│       └── app/
│           ├── main.py          # FastAPI app, model loading, inference routing
│           ├── database.py      # PostgreSQL connection + schema init
│           └── models/          # .pt / .pth weight files (not tracked by git)
├── test.py                      # Local integration test script
├── vis.py                       # Local visualization script (OpenCV)
└── README.md
```

> **Note:** Model weight files are excluded from version control via `.gitignore`. See [Quick Start](#quick-start) for setup.

---

## Database Schema

Detection records are stored in PostgreSQL with the following schema:

```sql
CREATE TABLE detections (
    id           SERIAL PRIMARY KEY,
    frame_id     VARCHAR(64),       -- UUID per inference call
    model_name   VARCHAR(64),       -- yolov5 | yolov8 | faster_rcnn
    label        VARCHAR(64),       -- detected class name
    confidence   FLOAT,             -- 0.0 – 1.0
    bbox         JSONB,             -- {x1, y1, x2, y2}
    drone_id     VARCHAR(64),       -- camera/drone identifier
    stream_url   TEXT,              -- RTSP source (future)
    detected_at  TIMESTAMP DEFAULT NOW()
);
```

---

## Roadmap

### Phase 1 — Core Inference ✅
- [x] Multi-model inference API (YOLOv5, YOLOv8, Faster R-CNN)
- [x] PostgreSQL detection persistence
- [x] GPU-accelerated Docker deployment

### Phase 2 — Streaming & Auth 🔧
- [ ] RTSP / HTTP stream ingestion pipeline
- [ ] JWT-based authentication service
- [ ] Per-drone session management

### Phase 3 — Frontend Dashboard 🔧
- [ ] React + Vite real-time visualization dashboard
- [ ] Dynamic model switching UI
- [ ] Confidence threshold and class filter controls
- [ ] Historical detection analytics charts

### Phase 4 — Advanced AI 📋
- [ ] SAHI integration for small object detection
- [ ] Grad-CAM / XAI visualization overlay service
- [ ] Async batch inference endpoint
- [ ] Model performance benchmarking panel

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/stream-ingestion`
3. Commit your changes: `git commit -m 'feat: add RTSP stream handler'`
4. Push to the branch: `git push origin feature/stream-ingestion`
5. Open a Pull Request

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---
