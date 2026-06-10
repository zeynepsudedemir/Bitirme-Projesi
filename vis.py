import subprocess
import json
import cv2
import ast


image_path = r"C:\Users\HP\Desktop\VisDrone\images\test\9999947_00000_d_0000023.jpg"
subprocess.run(["docker", "cp", image_path, "drone_inference:/tmp/test.jpg"])


result = subprocess.run([
    "docker", "exec", "drone_inference",
    "python3", "-c", """
import requests, json
resp = requests.post(
    'http://localhost:8001/api/v1/infer/sync',
    files={'file': open('/tmp/test.jpg', 'rb')},
    data={'model_name': 'faster_rcnn'}
)
print(json.dumps(resp.json()))
"""
], capture_output=True, text=True)

data = json.loads(result.stdout)
detections = data["detections"]


img = cv2.imread(image_path)

colors = {
    "pedestrian": (0, 255, 0),
    "people": (0, 200, 0),
    "bicycle": (255, 165, 0),
    "car": (0, 0, 255),
    "van": (255, 0, 255),
    "truck": (255, 0, 0),
    "tricycle": (0, 255, 255),
    "awning-tricycle": (0, 200, 200),
    "bus": (128, 0, 255),
    "motor": (255, 255, 0),
}

for det in detections:
    label = det["label"]
    conf = det["confidence"]
    bbox = det["bbox"]
    color = colors.get(label, (255, 255, 255))

    cv2.rectangle(img,
        (bbox["x1"], bbox["y1"]),
        (bbox["x2"], bbox["y2"]),
        color, 2
    )
    cv2.putText(img,
        f"{label} {conf:.2f}",
        (bbox["x1"], bbox["y1"] - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
    )

output_path = r"C:\Users\HP\Desktop\result3.jpg"
cv2.imwrite(output_path, img)
print(f"Kaydedildi: {output_path}")
print(f"Toplam tespit: {len(detections)}")