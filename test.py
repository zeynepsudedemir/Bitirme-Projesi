import urllib.request
import json

# Test görüntüsünü container'a kopyala
import subprocess
subprocess.run([
    "docker", "cp",
    r"C:\Users\HP\Desktop\VisDrone\images\test\9999947_00000_d_0000023.jpg",
    "drone_inference:/tmp/test.jpg"
])

# Container içinde test et
result = subprocess.run([
    "docker", "exec", "drone_inference",
    "python3", "-c", """
import requests
resp = requests.post(
    'http://localhost:8001/api/v1/infer/sync',
    files={'file': open('/tmp/test.jpg', 'rb')},
    data={'model_name': 'faster_rcnn'}
)
print(resp.json())
"""
], capture_output=True, text=True)

print(result.stdout)
print(result.stderr)