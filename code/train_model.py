from roboflow import Roboflow
from ultralytics import YOLO
import os

def train():
    # 1. הורדת הדאטהסט המעודכן (v2 - עם הדאטה הסינתטי)
    rf = Roboflow(api_key="wQZahQEnO6FVBKvlZq40")
    project = rf.workspace("none-dhjtb").project("roadside-parking-slots-2dpia")
    version = project.version(2)
    dataset = version.download("yolov11")

    # 2. טעינת המודל (YOLO11)
    model = YOLO('yolo11n.pt')

    # 3. אימון המודל על ה-RTX 4070
    results = model.train(
        data=os.path.join(dataset.location, "data.yaml"),
        epochs=100,           # כפי שהוגדר בתוכנית הפרויקט
        imgsz=640,
        device=0,             # שימוש ב-GPU
        batch=16,
        name='parkscope_final_model'
    )

if __name__ == '__main__':
    train()