from ultralytics import YOLO

model = YOLO("best .pt")

import os
print("Current working directory:", os.getcwd())

metrics = model.val(data="data.yaml")