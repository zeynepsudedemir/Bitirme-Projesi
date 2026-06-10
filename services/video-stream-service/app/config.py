
import os

class Settings:
    STREAM_SOURCE:str=os.getenv("STREAM_SOURCE", "0")  #webcam
    FRAME_SKIP:int=int(os.getenv("FRAME_SKIP", "1"))  # her n. frame
    MODEL_NAME:str=os.getenv("MODEL_NAME", "yolov8")  # varsayılan model
    CONFIDENCE_THRESHOLD:float=float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))  # varsayılan eşik değeri

    INFERENCE_SERVICE_URL:str=os.getenv(
        "INFERENCE_SERVICE_URL", 
        "http://inference-service:8001/api/v1/infer/sync"
    )  # varsayılan inference servisi URL'si

    INFERENCE_TIMEOUT:int=int(os.getenv("INFERENCE_TIMEOUT", "10"))  # varsayılan inference zaman aşımı (saniye)

    HOST:str=os.getenv("HOST","0.0.0.0")
    PORT:int=int(os.getenv("PORT", "8002"))

settings = Settings()
