import os
import cv2
import time
import tempfile
import torch

_sahi_models = {}


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_sahi_model(model_name: str):
    from sahi import AutoDetectionModel

    if model_name not in _sahi_models:
        path = f"app/models/{model_name}.pt"
        device = _get_device()
        model_type = "yolov8"  # SAHI, modeli otomatik algılayabiliyor ama bazen hata yapabiliyor
        _sahi_models[model_name] = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=path,
            confidence_threshold=0.4,
            device=device,
        )
    return _sahi_models[model_name]
def run_frcnn_sahi_inference(
    model,
    frame,          # np.ndarray, BGR
    frame_id: str,
    slice_size: int = 512,
    overlap_ratio: float = 0.2,
    conf_threshold: float = 0.5,
) -> tuple[list, int]:
    import time
    import torch
    import torchvision.transforms as T
    import torchvision.ops as ops

    FASTER_RCNN_LABELS = {
        1: "pedestrian", 2: "people", 3: "bicycle", 4: "car",
        5: "van", 6: "truck", 7: "tricycle", 8: "awning-tricycle",
        9: "bus", 10: "motor"
    }

    device = next(model.parameters()).device
    to_tensor = T.ToTensor()
    h, w = frame.shape[:2]
    step = int(slice_size * (1 - overlap_ratio))

    all_boxes = []
    all_scores = []
    all_labels = []

    t0 = time.perf_counter()

    for y in range(0, h, step):
        for x in range(0, w, step):
            x2 = min(x + slice_size, w)
            y2 = min(y + slice_size, h)
            slice_bgr = frame[y:y2, x:x2]

            import cv2
            slice_rgb = cv2.cvtColor(slice_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = to_tensor(slice_rgb).to(device)

            with torch.no_grad():
                output = model([img_tensor])[0]

            for box, label, score in zip(
                output["boxes"], output["labels"], output["scores"]
            ):
                if score < conf_threshold:
                    continue
                # Koordinatları orijinal görüntüye taşı
                all_boxes.append([
                    float(box[0]) + x,
                    float(box[1]) + y,
                    float(box[2]) + x,
                    float(box[3]) + y,
                ])
                all_scores.append(float(score))
                all_labels.append(int(label))

    inference_ms = int((time.perf_counter() - t0) * 1000)

    if not all_boxes:
        return [], inference_ms

    # NMS — örtüşen box'ları temizle
    boxes_tensor = torch.tensor(all_boxes)
    scores_tensor = torch.tensor(all_scores)
    keep = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)

    detections = []
    for i in keep.tolist():
        detections.append({
            "label": FASTER_RCNN_LABELS.get(all_labels[i], f"unknown_{all_labels[i]}"),
            "confidence": round(all_scores[i], 3),
            "bbox": {
                "x1": int(all_boxes[i][0]),
                "y1": int(all_boxes[i][1]),
                "x2": int(all_boxes[i][2]),
                "y2": int(all_boxes[i][3]),
            }
        })

    print(f"[SAHI FRCNN] inference={inference_ms}ms | "
          f"slices={(h // step + 1) * (w // step + 1)} | "
          f"detections={len(detections)}", flush=True)

    return detections, inference_ms

def run_sahi_inference(
    model_name: str,
    frame,          # np.ndarray, BGR
    frame_id: str,
    slice_size: int = 512,
    overlap_ratio: float = 0.2,
) -> tuple[list, int]:
    """
    Returns: (detections: list[dict], inference_ms: int)
    """
    from sahi.predict import get_sliced_prediction

    model = get_sahi_model(model_name)

    # Geçici dosyayı context manager ile yönet → exception'da da silinir
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, frame)

        t0 = time.perf_counter()
        result = get_sliced_prediction(
            tmp_path,
            model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            verbose=0,          # log gürültüsünü kapat
            perform_standard_pred=True,   # tam görüntüde de bir kez çalıştır
        )
        inference_ms = int((time.perf_counter() - t0) * 1000)
        print(f"[SAHI] {model_name} inference={inference_ms}ms | "
              f"detections={len(result.object_prediction_list)}", flush=True)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    detections = []
    for obj in result.object_prediction_list:
        bbox = obj.bbox
        detections.append({
            "label": obj.category.name,
            "confidence": round(obj.score.value, 3),
            "bbox": {
                "x1": int(bbox.minx),
                "y1": int(bbox.miny),
                "x2": int(bbox.maxx),
                "y2": int(bbox.maxy),
            },
        })

    return detections, inference_ms