# API endpoint tanımları — stream başlatma/durdurma + WebSocket
import asyncio
import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from app.stream_manager import stream_manager, ws_manager, StreamStatus
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stream Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartStreamRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    drone_id: str
    source: Optional[str] = None
    model_name: Optional[str] = None
    confidence_threshold: Optional[float] = None


class UpdateModelRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str


# endpointler

@app.get("/health")
def health():
    streams = stream_manager.list_streams()
    return {
        "status": "ok",
        "active_streams": len(streams),
        "inference_service_url": settings.INFERENCE_SERVICE_URL,
    }


@app.post("/streams", status_code=201)
def start_stream(req: StartStreamRequest):
    try:
        info = stream_manager.start_stream(
            drone_id=req.drone_id,
            source=req.source,
            model_name=req.model_name,
            confidence_threshold=req.confidence_threshold,
        )
        return {"message": f"Stream başlatıldı: {req.drone_id}", "stream": info.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"[API ERROR] start_stream: {e}")
        raise HTTPException(status_code=500, detail="Stream başlatılamadı")


@app.get("/streams")
def list_streams():
    streams = stream_manager.list_streams()
    return {"total": len(streams), "streams": streams}


@app.get("/streams/{drone_id}")
def get_stream(drone_id: str):
    info = stream_manager.get_stream(drone_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Stream bulunamadı: {drone_id}")
    return info.to_dict()


@app.delete("/streams/{drone_id}")
def stop_stream(drone_id: str):
    try:
        stream_manager.stop_stream(drone_id)
        return {"message": f"Stream durduruldu: {drone_id}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"[API ERROR] stop_stream: {e}")
        raise HTTPException(status_code=500, detail="Stream durdurulamadı")

stream_models: dict = {}
@app.patch("/streams/{drone_id}/model")
def update_model(drone_id: str, req: UpdateModelRequest):
    valid_models = ["yolov5", "yolov8", "faster_rcnn"]
    if req.model_name not in valid_models:
        raise HTTPException(status_code=400, detail=f"Geçersiz model. Seçenekler: {valid_models}")
    try:
        stream_manager.change_model(drone_id, req.model_name)
        return {"message": f"Model güncellendi: {drone_id}", "new_model": req.model_name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"[API ERROR] update_model: {e}")
        raise HTTPException(status_code=500, detail="Model güncellenemedi")


#WebSocket bağlantısı

@app.websocket("/ws/{drone_id}")
async def websocket_endpoint(websocket: WebSocket, drone_id: str):
    """
    Frontend buraya bağlanır.
    Stream worker her frame sonucunu bu socket üzerinden iletir.

    Mesaj formatı:
    {
        "drone_id":    "cam1",
        "frame_index": 42,
        "model_name":  "yolov8",
        "detections": [
            {"label": "car", "confidence": 0.91, "bbox": {"x1":10,"y1":20,"x2":80,"y2":90}},
            ...
        ]
    }
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()
    ws_manager.add(drone_id, (websocket, loop))
    logger.info(f"[WS] Bağlandı → drone_id={drone_id}")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"[WS] Ayrıldı → drone_id={drone_id}")
    finally:
        ws_manager.remove(drone_id, (websocket, loop))