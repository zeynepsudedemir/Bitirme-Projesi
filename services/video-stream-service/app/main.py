# API endpoint tanımları stream başlatma durdurma
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from app.stream_manager import stream_manager, StreamStatus
from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stream Service", version="0.1.0")


class StartStreamRequest(BaseModel):
    drone_id: str
    source: Optional[str] = None              # None → config'den alır
    model_name: Optional[str] = None          # None → config'den alır
    confidence_threshold: Optional[float] = None  # None → config'den alır


class UpdateModelRequest(BaseModel):
    model_name: str                            # yolov5 | yolov8 | faster_rcnn


@app.get("/health")
def health():
    """Servis ayakta mı? Aktif stream sayısını döner."""
    streams = stream_manager.list_streams()
    return {
        "status": "ok",
        "active_streams": len(streams),
        "inference_service_url": settings.INFERENCE_SERVICE_URL,
    }


@app.post("/streams")
def start_stream(req: StartStreamRequest):
    """
    Yeni bir stream başlatır.
    - drone_id: Kameranın benzersiz kimliği
    - source: "0" (webcam), "rtsp://...", "video.mp4"
    - model_name: yolov5 | yolov8 | faster_rcnn
    - confidence_threshold: 0.0 - 1.0
    """
    try:
        info = stream_manager.start_stream(
            drone_id=req.drone_id,
            source=req.source,
            model_name=req.model_name,
            confidence_threshold=req.confidence_threshold,
        )
        return JSONResponse(
            status_code=201,
            content={
                "message": f"Stream başlatıldı: {req.drone_id}",
                "stream": info.to_dict(),
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"[API ERROR] start_stream: {e}")
        raise HTTPException(status_code=500, detail="Stream başlatılamadı")


@app.get("/streams")
def list_streams():
    """Tüm aktif stream'leri listeler."""
    streams = stream_manager.list_streams()
    return {
        "total": len(streams),
        "streams": streams,
    }


@app.get("/streams/{drone_id}")
def get_stream(drone_id: str):
    """Belirli bir stream'in durumunu döner."""
    info = stream_manager.get_stream(drone_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Stream bulunamadı: {drone_id}")
    return info.to_dict()


@app.delete("/streams/{drone_id}")
def stop_stream(drone_id: str):
    """Çalışan bir stream'i durdurur."""
    try:
        stream_manager.stop_stream(drone_id)
        return {"message": f"Stream durduruldu: {drone_id}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"[API ERROR] stop_stream: {e}")
        raise HTTPException(status_code=500, detail="Stream durdurulamadı")


@app.patch("/streams/{drone_id}/model")
def update_model(drone_id: str, req: UpdateModelRequest):
    """
    Stream'i durdurmadan model değiştirir.
    Dashboard'dan canlı model switching için kullanılacak.
    """
    valid_models = ["yolov5", "yolov8", "faster_rcnn"]
    if req.model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz model. Seçenekler: {valid_models}"
        )
    try:
        stream_manager.update_model(drone_id, req.model_name)
        return {
            "message": f"Model güncellendi: {drone_id}",
            "new_model": req.model_name,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"[API ERROR] update_model: {e}")
        raise HTTPException(status_code=500, detail="Model güncellenemedi")