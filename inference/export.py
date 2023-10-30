from ultralytics import YOLO
import shutil
import os

os.chdir("inference")

MODEL_NAME = "yolov8n"
TASK = "detect"
IMAGE_SIZE = 640
SAVED_MODEL_NAME = "model"
SAVED_MODEL_FORMAT = "onnx"
SAVED_MODEL_FOLDER = "inference"
SAVED_MODEL_PATH = f"{SAVED_MODEL_FOLDER}/{SAVED_MODEL_NAME}.{SAVED_MODEL_FORMAT}"

model = YOLO(model=f"{MODEL_NAME}.pt", task=TASK)
model.export(format=SAVED_MODEL_FORMAT, imgsz=IMAGE_SIZE)

shutil.move(f"{MODEL_NAME}.{SAVED_MODEL_FORMAT}", SAVED_MODEL_PATH)
os.remove(f"{MODEL_NAME}.pt")
