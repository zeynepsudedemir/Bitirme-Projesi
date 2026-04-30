from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.database import init_db, get_connection
import torch
import uuid
import cv2
import numpy as np
from ultralytics import YOLO

app=FastAPI(title="Inference Service",version="0.1.0")

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
        sizes=((32,), (64,), (128,), (256,), (512,)),
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

@app.post("/api/v1/infer/sync")
async def infer_sync(file: UploadFile,model_name:str="yolov8"):
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Model not found")
    
    contents =await file.read()
    np_arr =np.frombuffer(contents, np.uint8)
    frame= cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Görüntü okunamadı")
    
    frame_id=str(uuid.uuid4())
    detections=_run_inference(model_name,frame,frame_id)

    return JSONResponse({
        "frame_id": frame_id,
        "model_name": model_name,
        "detections": detections
    })

def _run_inference(model_name:str, frame: np.ndarray, frame_id:str):
    detections=[]
    if model_name == "yolov5":
        results = models["yolov5"](frame)
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
                _save_detection(frame_id, model_name, det)

    elif model_name=="yolov8":
        results = models["yolov8"](frame, conf=0.4)
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
                _save_detection(frame_id, model_name, det)
    elif model_name=="faster_rcnn":
        import torchvision.transforms as T
        transform = T.ToTensor()
        img_tensor = transform(frame).unsqueeze(0)
        device = next(models["faster_rcnn"].parameters()).device
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            outputs = models["faster_rcnn"](img_tensor)[0]

        for box, label, score in zip(
            outputs["boxes"], outputs["labels"], outputs["scores"]
        ):
            if score < 0.5:
                continue
            det = {
                "label": str(int(label)),
                "confidence": round(float(score), 3),
                "bbox": {
                    "x1": int(box[0]), "y1": int(box[1]),
                    "x2": int(box[2]), "y2": int(box[3])
                }
            }
            detections.append(det)
            _save_detection(frame_id, model_name, det)

    return detections

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