from ultralytics import YOLO

model = YOLO("best .pt")

import os
print("Current working directory:", os.getcwd())

map50 = 0.629
map5095 = 0.485
precision = 0.737
recall = 0.548
