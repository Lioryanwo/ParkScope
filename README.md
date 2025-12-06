🚘 ParkScope – AI-Based Free Street Parking Spot Detection
Object Detection Using YOLO + Inpainting
⭐ Overview

Finding street parking in dense urban environments is a daily challenge. ParkScope proposes a vision-based system that detects free parking spots from a single street-level image — without sensors, without historical data, and without manual labeling.

The system relies on:

Car detection

Diffusion-based inpainting to remove vehicles

Automatic generation of parking-spot labels

Training a YOLO model to detect free parking spaces

🎯 Project Goal

Train an object detection model that identifies free parking spots in real street images.

The model outputs:

Bounding boxes corresponding to empty parking spots

🧩 System Pipeline
Street Image → Car Detection (YOLO) → Inpainting → Auto-Labeling → Train YOLO → Predict Free Spots

📚 Dataset

ParkScope uses two types of data:

1️⃣ Real Images

KITTI Dataset – real street-level images captured from a car-mounted camera.

These serve as the foundation for generating labeled training samples.

2️⃣ Automatically Generated Training Data

Created via a custom labeling pipeline:

YOLO detects all cars in the scene.

A diffusion-based inpainting model removes the cars.

The reconstructed sidewalk/curb reveals potential free parking areas.

These areas are converted into bounding-box labels for “Free Parking Spot.”

This enables large-scale dataset creation without manual annotation.

🏗️ Data Generation Pipeline
✔️ Car Detection

YOLO extracts bounding boxes around all visible vehicles.

✔️ Inpainting

A diffusion-based inpainting model removes each detected vehicle and realistically reconstructs the background curb/road.

✔️ Auto-Labeling

The newly exposed regions along the curb are interpreted and saved as:

Free Parking Spot

This produces fully supervised training samples automatically.

🔧 Data Augmentation

To improve robustness and model generalization:

Lighting variations (day/evening/shadows)

Brightness & contrast shifts

Noise, blur, weather-like distortions

Random cropping

🧠 Model
✔️ YOLO (v8 / v9) – Free Parking Spot Detector

This is the main and only model used in the project.
It is trained to detect free parking spaces using the auto-generated dataset.

🏋️ Training Process

Build a complete training set from:

Inpainted images

Automatically generated bounding boxes

Split into:

Train

Validation

Test

Train YOLO for the task of free-spot detection.

Evaluate, refine augmentations, and improve the dataset.

📏 Evaluation Metrics

The model is evaluated using:

mAP@50

Precision / Recall

Qualitative visualization examples

📈 Results

(To be added: Inpainting samples, YOLO predictions, training curves, quantitative tables.)

📁 Repository Structure

(Can be added once the folder tree is finalized. I can generate it for you.)

👤 Team

Lior Yanwo
