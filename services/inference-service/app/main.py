import json

from fastapi.middleware.cors import CORSMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware 
from app.database import init_db, get_connection
import torch
import uuid
import cv2
import numpy as np
from ultralytics import YOLO
import time
app=FastAPI(title="Inference Service",version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models={}

@app.on_event("startup")
def startup():
    init_db()
    _load_models()

def _load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Modeller yükleniyor — device: {device}")

    models["yolov5"] =YOLO("app/models/yolov5.pt")
    models["yolov5"].to(device)

    models["yolov8"]=YOLO("app/models/yolov8.pt")
    models["yolov8"].to(device)

    import torchvision
    checkpoint = torch.load(
        "app/models/faster_rcnn.pth",
        map_location=device,
        weights_only=False
    )

    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.rpn import AnchorGenerator

    backbone = resnet_fpn_backbone("resnet101", weights=None)

    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0, 0.25, 1.5),) * 5
    )

    frcnn = FasterRCNN(
        backbone,
        num_classes=11,
        rpn_anchor_generator=anchor_generator
    )
    frcnn.load_state_dict(checkpoint["model_state_dict"])
    frcnn.to(device)
    frcnn.eval()
    models["faster_rcnn"] = frcnn

    print("[INFO] 3 model yüklendi.")

@app.get("/health")
def health():
    gpu=_check_gpu()
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": list(models.keys())
    }

executor = ThreadPoolExecutor(max_workers=2)

@app.get("/api/v1/models/metrics")
def get_metrics():
    with open("app/models/metrics.json", "r") as f:
        return json.load(f)


import base64
from app.gradcam import gradcam_yolo, gradcam_faster_rcnn

@app.post("/api/v1/infer/gradcam")
async def infer_gradcam(file: UploadFile, model_name: str = "yolov8"):
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Model not found")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Görüntü okunamadı")

    frame_id = str(uuid.uuid4())

    # Normal inference
    t0 = time.perf_counter()
    detections, inference_ms = _run_inference(model_name, frame, frame_id)
    inference_ms = int((time.perf_counter() - t0) * 1000)

    # GradCAM with detections highlighted in RED
    heatmap_b64 = None
    try:
        if model_name in ("yolov8", "yolov5"):
            heatmap = gradcam_yolo(models[model_name], frame, detections=detections)
        else:
            heatmap = gradcam_faster_rcnn(models["faster_rcnn"], frame, detections=detections)

        # Heatmap to base64
        if heatmap is not None and heatmap.size > 0:
            _, buf = cv2.imencode(".jpg", heatmap, [cv2.IMWRITE_JPEG_QUALITY, 85])
            heatmap_b64 = base64.b64encode(buf).decode("utf-8")
    except Exception as e:
        print(f"[GRADCAM ERROR] {e}")
        heatmap_b64 = None

    return JSONResponse({
        "frame_id": frame_id,
        "model_name": model_name,
        "detections": detections,
        "inference_ms": inference_ms,
        "heatmap": heatmap_b64
    }) 

@app.post("/api/v1/infer/sync")
async def infer_sync(file: UploadFile, model_name: str = "yolov8", conf_threshold: float = None):
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Model not found")
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    print(f"[DEBUG] frame shape: {frame.shape}", flush=True)
    if frame is None:
        raise HTTPException(status_code=400, detail="Görüntü okunamadı")
    frame_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    start = time.perf_counter()
    detections, inference_ms = _run_inference(model_name, frame, frame_id)
    return JSONResponse({"frame_id": frame_id, "model_name": model_name, "detections": detections, "inference_ms": inference_ms})

FASTER_RCNN_LABELS = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor"
}
CONF_THRESHOLDS = {
    "yolov5": 0.4,
    "yolov8": 0.4,
    "faster_rcnn": 0.3
}

import torchvision.transforms as T
_to_tensor = T.ToTensor()

def _run_inference(model_name:str, frame: np.ndarray, frame_id:str, conf_threshold: float = None):

    threshold = conf_threshold or CONF_THRESHOLDS.get(model_name, 0.4)
    detections = []
    inference_ms = 0
    import time

    if model_name == "yolov5":
        t0 = time.perf_counter()
        results = models["yolov5"](frame, conf=threshold,imgsz=640)
        inference_ms = int((time.perf_counter() - t0) * 1000)
        print(f"[TIMING] yolov5 inference={inference_ms}ms", flush=True)
        for r in results:
            for box in r.boxes:
                label = models["yolov5"].names[int(box.cls)]
                det = {
                    "label": label,
                    "confidence": round(float(box.conf), 3),
                    "bbox": {
                        "x1": int(box.xyxy[0][0]), "y1": int(box.xyxy[0][1]),
                        "x2": int(box.xyxy[0][2]), "y2": int(box.xyxy[0][3])
                    }
                }
                detections.append(det)

    elif model_name == "yolov8":
        t0 = time.perf_counter()
        results = models["yolov8"](frame, conf=threshold,imgsz=640)
        inference_ms = int((time.perf_counter() - t0) * 1000)
        print(f"[TIMING] yolov8 inference={inference_ms}ms", flush=True)
        for r in results:
            for box in r.boxes:
                label = models["yolov8"].names[int(box.cls)]
                det = {
                    "label": label,
                    "confidence": round(float(box.conf), 3),
                    "bbox": {
                        "x1": int(box.xyxy[0][0]), "y1": int(box.xyxy[0][1]),
                        "x2": int(box.xyxy[0][2]), "y2": int(box.xyxy[0][3])
                    }
                }
                detections.append(det)

    elif model_name == "faster_rcnn":
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        device = next(models["faster_rcnn"].parameters()).device
        img_tensor = _to_tensor(frame_rgb).unsqueeze(0).to(device)
        t0 = time.perf_counter()  # sadece forward pass
        with torch.no_grad():
            outputs = models["faster_rcnn"](img_tensor)[0]
        inference_ms = int((time.perf_counter() - t0) * 1000)
        
        print(f"[TIMING] faster_rcnn inference={inference_ms}ms", flush=True)

        for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
            if score < threshold:
                continue
            det = {
                "label": FASTER_RCNN_LABELS.get(int(label), f"Unknown {label}"),
                "confidence": round(float(score), 3),
                "bbox": {
                    "x1": int(box[0]), "y1": int(box[1]),
                    "x2": int(box[2]), "y2": int(box[3])
                }
            }
            detections.append(det)

    return detections, inference_ms

def _save_detection(frame_id: str, model_name: str, det: dict):
    try:
        import json
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO detections
                (frame_id, model_name, label, confidence, bbox)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            frame_id,
            model_name,
            det["label"],
            det["confidence"],
            json.dumps(det["bbox"])
        ))
        conn.commit()
        cur.close()
        conn.close()
        print(f"[DB] Kaydedildi: {det['label']} - {det['confidence']}")
    except Exception as e:
        print(f"[DB ERROR] {e}")


def _check_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False