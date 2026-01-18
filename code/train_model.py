from ultralytics import YOLO
import os

def train():
    # 1. Reference the local dataset path instead of downloading every time
    # The folder name should match what Roboflow created in your project directory
    dataset_path = os.path.join('Roadside-Parking-Slots-1', 'data.yaml')

    # Verify if dataset exists before starting
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Please run download_dataset.py first!")
        return

    # 2. Loading the Model (YOLO11)
    model = YOLO('yolo11n.pt')

    # 3. Model Training on NVIDIA GPU
    results = model.train(
        data=dataset_path,
        epochs=100,           
        imgsz=640,
        device=0, # Ensuring GPU usage for high-performance training
        batch=16,
        name='parkscope_final_model'
    )

if __name__ == '__main__':
    train()