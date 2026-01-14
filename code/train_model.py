from roboflow import Roboflow
from ultralytics import YOLO
import os

def train():
    # 1. Downloading Updated Dataset (v2 - including synthetic data)
    rf = Roboflow(api_key="wQZahQEnO6FVBKvlZq40")
    project = rf.workspace("none-dhjtb").project("roadside-parking-slots-2dpia")
    version = project.version(2)
    dataset = version.download("yolov11")

    # 2. Loading the Model (YOLO11)
    model = YOLO('yolo11n.pt')

    # 3. Model Training on NVIDIA GPU
    results = model.train(
        data=os.path.join(dataset.location, "data.yaml"),
        epochs=100,           
        imgsz=640,
        device=0,             
        batch=16,
        name='parkscope_final_model'
    )

if __name__ == '__main__':
    train()