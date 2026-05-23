import json
import time
import base64
import uuid

from fastapi.middleware.cors import CORSMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as T

app = FastAPI(title="Inference Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {}

_to_tensor = T.ToTensor()

executor = ThreadPoolExecutor(max_workers=4)

_gradcam_cache: dict = {}      
GRADCAM_TTL_SEC = 2.0           # 2 saniyede yenile 

FASTER_RCNN_LABELS = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car",
    5: "van", 6: "truck", 7: "tricycle", 8: "awning-tricycle",
    9: "bus", 10: "motor"
}

CONF_THRESHOLDS = {
    "yolov5": 0.5,     
    "yolov8": 0.5,      
    "faster_rcnn": 0.4  
}


TRY_SAHI = True
try:
    from app.sahi_run import run_sahi_inference
    from sahi import AutoDetectionModel  # noqa
except (ImportError, ModuleNotFoundError) as e:
    print(f"[WARN] SAHI devre dışı: {e}")
    TRY_SAHI = False
    run_sahi_inference = None


@app.on_event("startup")
def startup():
    from app.database import init_db
    init_db()
    _load_models()


def _load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Modeller yükleniyor — device: {device}")

    # ── YOLOv5 ───────────────────────────────────────────────────────────────
    models["yolov5"] = YOLO("app/models/yolov5.pt")
    models["yolov5"].to(device)
    # Warmup 
    _warmup_yolo(models["yolov5"], device)

    # ── YOLOv8 ───────────────────────────────────────────────────────────────
    models["yolov8"] = YOLO("app/models/yolov8.pt")
    models["yolov8"].to(device)
    _warmup_yolo(models["yolov8"], device)

    # ── Faster R-CNN ─────────────────────────────────────────────────────────
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.rpn import AnchorGenerator

    checkpoint = torch.load(
        "app/models/faster_rcnn.pth",
        map_location=device,
        weights_only=False
    )
    backbone = resnet_fpn_backbone("resnet101", weights=None)
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0, 0.25, 1.5),) * 5
    )
    frcnn = FasterRCNN(backbone, num_classes=11, rpn_anchor_generator=anchor_generator)
    frcnn.load_state_dict(checkpoint["model_state_dict"])
    frcnn.to(device)
    frcnn.eval()

    # Faster R-CNN warmup
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 480, 640).to(device)
        frcnn([dummy[0]])

    models["faster_rcnn"] = frcnn
    print("[INFO] 3 model yüklendi + warmup tamamlandı.")


def _warmup_yolo(model, device):
    """İlk CUDA kernel launch gecikmesini startup'ta öde."""
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False, imgsz=640)
    print(f"[INFO] Warmup tamamlandı: {model}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": list(models.keys())
    }


@app.get("/api/v1/models/metrics")
def get_metrics():
    with open("app/models/metrics.json", "r") as f:
        return json.load(f)


from app.gradcam import gradcam_yolo, gradcam_faster_rcnn


def _get_cached_gradcam(model_name: str, frame: np.ndarray) -> str | None:

    now = time.monotonic()
    cached = _gradcam_cache.get(model_name)
    if cached and (now - cached["ts"]) < GRADCAM_TTL_SEC:
        return cached["b64"]

    try:
        if model_name in ("yolov8", "yolov5"):
            heatmap = gradcam_yolo(models[model_name], frame)
        else:
            heatmap = gradcam_faster_rcnn(models["faster_rcnn"], frame)

        if heatmap is not None and heatmap.size > 0:
            _, buf = cv2.imencode(".jpg", heatmap, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buf).decode("utf-8")
            _gradcam_cache[model_name] = {"b64": b64, "ts": now}
            return b64
    except Exception as e:
        print(f"[GRADCAM ERROR] {e}")
    return None


@app.post("/api/v1/infer/gradcam")
async def infer_gradcam(file: UploadFile, model_name: str = "yolov8", enable_gradcam: bool = True):
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Model not found")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Görüntü okunamadı")

    frame_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    detections, inference_ms = _run_inference(model_name, frame, frame_id)
    inference_ms = int((time.perf_counter() - t0) * 1000)

    heatmap_b64 = _get_cached_gradcam(model_name, frame) if enable_gradcam else None

    return JSONResponse({
        "frame_id": frame_id,
        "model_name": model_name,
        "detections": detections,
        "inference_ms": inference_ms,
        "heatmap": heatmap_b64
    })


@app.post("/api/v1/infer/sync")
async def infer_sync(
    file: UploadFile,
    model_name: str = "yolov8",
    use_sahi: bool = False,
    slice_size: int = 640,
    overlap_ratio: float = 0.2,
    enable_gradcam: bool = False
):
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Model not found")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Görüntü okunamadı")

    print(f"[DEBUG] frame shape: {frame.shape} | model: {model_name} | sahi: {use_sahi} | gradcam: {enable_gradcam}", flush=True)

    frame_id = str(uuid.uuid4())

    # SAHI mode: kurulu ise ve isteniyorsa YOLO modelleri için kullan
    if use_sahi and model_name == "faster_rcnn":
        try:
            from app.sahi_run import run_frcnn_sahi_inference
            detections, inference_ms = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: run_frcnn_sahi_inference(
                    models["faster_rcnn"], frame, frame_id,
                    slice_size, overlap_ratio
                )
            )
        except Exception as e:
            print(f"[WARN] FRCNN SAHI başarısız ({e}), normal inference'a geçildi")
            detections, inference_ms = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: _run_inference(model_name, frame, frame_id)
            )
    elif use_sahi and TRY_SAHI and model_name in ("yolov5", "yolov8"):
        try:
            detections, inference_ms = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: run_sahi_inference(model_name, frame, frame_id, slice_size, overlap_ratio)
            )
        except Exception as e:
            print(f"[WARN] SAHI inference başarısız ({e}), normal inference'a geçildi", flush=True)
            detections, inference_ms = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: _run_inference(model_name, frame, frame_id)
            )
    else:
        if use_sahi and not TRY_SAHI:
            print("[WARN] SAHI kurulu değil, normal inference çalıştırılıyor.", flush=True)
        if use_sahi and model_name == "faster_rcnn":
            print("[WARN] SAHI, Faster R-CNN ile desteklenmez. Normal inference çalıştırılıyor.", flush=True)
        detections, inference_ms = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: _run_inference(model_name, frame, frame_id)
        )

    # GradCAM: önbellekli — her frame'de tam hesaplama yapılmaz
    heatmap_b64 = None
    if enable_gradcam:
        heatmap_b64 = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: _get_cached_gradcam(model_name, frame)
        )

    return JSONResponse({
        "frame_id": frame_id,
        "model_name": model_name,
        "detections": detections,
        "inference_ms": inference_ms,
        "sahi_used": use_sahi and model_name in ("yolov5", "yolov8"),
        "heatmap": heatmap_b64
    })


def _run_inference(model_name: str, frame: np.ndarray, frame_id: str, conf_threshold: float = None):
    threshold = conf_threshold or CONF_THRESHOLDS.get(model_name, 0.4)
    detections = []

    if model_name in ("yolov5", "yolov8"):
        yolo_model = models[model_name]

        # ── Frame boyutunu modele göre AGRESIF optimize et ──────────────────
        # Daha küçük resolution = daha hızlı inference
        # YOLO imgsz: 32'nin katı olmalı, max 640, min 320
        h, w = frame.shape[:2]
        long_side = max(h, w)
        if long_side <= 360:
            imgsz = 320   # Çok küçük → 320
        elif long_side <= 480:
            imgsz = 416   # Orta → 416 (default, hızlı)
        elif long_side <= 600:
            imgsz = 480   # Biraz büyük → 480
        else:
            imgsz = 544   # Büyük → 544 (640'dan daha hızlı)

        t0 = time.perf_counter()
        results = yolo_model(
            frame,
            conf=threshold,
            imgsz=imgsz,
            verbose=False,           # Log baskı kapatıldı
            half=torch.cuda.is_available(),  # GPU varsa FP16
            device=0 if torch.cuda.is_available() else "cpu",  # Explicit device
        )
        inference_ms = int((time.perf_counter() - t0) * 1000)
        if inference_ms > 200:
            print(f"[TIMING] {model_name} SLOW inference={inference_ms}ms imgsz={imgsz}", flush=True)

        label_map = yolo_model.names
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "label": label_map[int(box.cls)],
                    "confidence": round(float(box.conf), 3),
                    "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
                })

    elif model_name == "faster_rcnn":
        device = next(models["faster_rcnn"].parameters()).device
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = _to_tensor(frame_rgb).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = models["faster_rcnn"](img_tensor)[0]
        inference_ms = int((time.perf_counter() - t0) * 1000)
        print(f"[TIMING] faster_rcnn inference={inference_ms}ms", flush=True)

        for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
            if score < threshold:
                continue
            detections.append({
                "label": FASTER_RCNN_LABELS.get(int(label), f"Unknown {label}"),
                "confidence": round(float(score), 3),
                "bbox": {
                    "x1": int(box[0]), "y1": int(box[1]),
                    "x2": int(box[2]), "y2": int(box[3])
                }
            })

    return detections, inference_ms


def _save_detection(frame_id: str, model_name: str, det: dict):
    try:
        from app.database import get_connection
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO detections (frame_id, model_name, label, confidence, bbox)
            VALUES (%s, %s, %s, %s, %s)
        """, (frame_id, model_name, det["label"], det["confidence"], json.dumps(det["bbox"])))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] {e}")


def _check_gpu():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False