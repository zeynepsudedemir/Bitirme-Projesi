# kamera okuma ve frame gönderme
import cv2
import threading
import requests
import logging
import time
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamStatus(str, Enum):
    STARTING = "starting"
    RUNNING  = "running"
    STOPPED  = "stopped"
    ERROR    = "error"


@dataclass
class StreamInfo:
    drone_id: str
    source: str
    model_name: str
    confidence_threshold: float
    status: StreamStatus = StreamStatus.STARTING
    frame_count: int = 0
    detection_count: int = 0
    error_message: Optional[str] = None
    _stop_event: threading.Event = field(default_factory=threading.Event)

    def to_dict(self) -> dict:
        return {
            "drone_id":            self.drone_id,
            "source":              self.source,
            "model_name":          self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "status":              self.status.value,
            "frame_count":         self.frame_count,
            "detection_count":     self.detection_count,
            "error_message":       self.error_message,
        }


#WebSocket bağlantı
class ConnectionManager:
    def __init__(self):
        self._clients: dict[str, set] = {}
        self._lock = threading.Lock()

    def add(self, drone_id: str, ws) -> None:
        with self._lock:
            self._clients.setdefault(drone_id, set()).add(ws)

    def remove(self, drone_id: str, ws) -> None:
        with self._lock:
            if drone_id in self._clients:
                self._clients[drone_id].discard(ws)

    def broadcast(self, drone_id: str, payload: dict) -> None:
        with self._lock:
            clients = list(self._clients.get(drone_id, []))
        for ws_tuple in clients:
            ws, loop = ws_tuple
            asyncio.run_coroutine_threadsafe(
                ws.send_json(payload), loop
            )


ws_manager = ConnectionManager()


class StreamManager:
    def __init__(self):
        self.streams: dict = {}
        self._lock = threading.Lock()

    def start_stream(
        self,
        drone_id: str,
        source: Optional[str] = None,
        model_name: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ) -> StreamInfo:

        with self._lock:
            if drone_id in self.streams:
                raise ValueError(f"Stream for drone_id {drone_id} already exists")

            info = StreamInfo(
                drone_id=drone_id,
                source=source or settings.STREAM_SOURCE,
                model_name=model_name or settings.MODEL_NAME,
                confidence_threshold=confidence_threshold or settings.CONFIDENCE_THRESHOLD,
            )
            self.streams[drone_id] = info

        thread = threading.Thread(
            target=self._worker,
            args=(info,),
            daemon=True,
            name=f"stream-{drone_id}",
        )
        thread.start()
        logger.info(f"[STREAM] Başlatıldı → drone_id={drone_id} source={info.source}")
        return info

    def stop_stream(self, drone_id: str) -> None:
        with self._lock:
            info = self.streams.get(drone_id)
            if info is None:
                raise ValueError(f"Stream for drone_id {drone_id} does not exist")
            info._stop_event.set()
            logger.info(f"[STREAM] Durduruluyor → drone_id={drone_id}")

    def get_stream(self, drone_id: str) -> Optional[StreamInfo]:
        return self.streams.get(drone_id)

    def list_streams(self) -> list[dict]:
        with self._lock:
            return [s.to_dict() for s in self.streams.values()]

    def change_model(self, drone_id: str, model_name: str) -> None:
        with self._lock:
            info = self.streams.get(drone_id)
            if info is None:
                raise ValueError(f"Stream for drone_id {drone_id} does not exist")
            info.model_name = model_name
            logger.info(f"[STREAM] Model değiştirildi → drone_id={drone_id} model={model_name}")

    #Worker
    def _worker(self, info: StreamInfo) -> None:
        source = int(info.source) if info.source.isdigit() else info.source
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            info.status = StreamStatus.ERROR
            info.error_message = f"Kamera/stream açılamadı: {info.source}"
            logger.error(f"[STREAM ERROR] {info.error_message}")
            return

        info.status = StreamStatus.RUNNING
        frame_index = 0

        try:
            while not info._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, str) and source.endswith(".mp4"):
                        logger.info(f"[STREAM] Video bitti → {info.drone_id}")
                        break
                    else:
                        logger.warning(f"[STREAM] Bağlantı koptu, yeniden bağlanılıyor → {info.drone_id}")
                        time.sleep(2)
                        cap.release()
                        cap = cv2.VideoCapture(source)
                    continue

                info.frame_count += 1
                frame_index += 1

                if frame_index % settings.FRAME_SKIP != 0:
                    continue

                detections = self._send_to_inference(frame, info)

                if detections is not None:
                    info.detection_count += len(detections)
                    # WebSocket istemcilerine gönder
                    ws_manager.broadcast(info.drone_id, {
                        "drone_id":    info.drone_id,
                        "frame_index": frame_index,
                        "model_name":  info.model_name,
                        "detections":  detections,
                    })

        except Exception as e:
            info.status = StreamStatus.ERROR
            info.error_message = str(e)
            logger.error(f"[STREAM ERROR] drone_id={info.drone_id} error={str(e)}")

        finally:
            cap.release()
            info.status = StreamStatus.STOPPED
            with self._lock:
                self.streams.pop(info.drone_id, None)
            logger.info(f"[STREAM] Kapandı → {info.drone_id}")

    #Inference servisine gönderme
    def _send_to_inference(self, frame, info: StreamInfo) -> Optional[list]:
        try:
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                logger.warning(f"[INFERENCE] Frame encode edilemedi → {info.drone_id}")
                return None

            files    = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
            data     = {"model_name": info.model_name}
            response = requests.post(
                settings.INFERENCE_SERVICE_URL,
                files=files,
                data=data,
                timeout=settings.INFERENCE_TIMEOUT,
            )
            response.raise_for_status()

            result     = response.json()
            detections = result.get("detections", [])
            detections = [d for d in detections if d["confidence"] >= info.confidence_threshold]

            logger.info(
                f"[INFER] {info.drone_id} | model={info.model_name} | tespit={len(detections)}"
            )
            return detections

        except requests.exceptions.Timeout:
            logger.warning(f"[INFER] Timeout → {info.drone_id}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"[INFER] Bağlantı hatası → {info.drone_id}")
        except Exception as e:
            logger.error(f"[INFER ERROR] {info.drone_id}: {e}")

        return None


stream_manager = StreamManager()