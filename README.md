🚘 ParkScope – AI-Based Parking Difficulty Prediction
⭐ Project Motivation

Finding street parking in dense urban areas often results in long search times, increased congestion, wasted fuel, and driver frustration.
Despite advanced navigation tools, no existing system (Waze, Google Maps) provides a real-time estimation of how difficult it will be to find parking near your destination.

ParkScope introduces an AI-driven parking difficulty estimator designed to enhance navigation and reduce unnecessary driving loops.

🎯 Problem Statement

Predict a Parking Difficulty Score (1–10) from a single street-level image, indicating how likely it is to find an available parking spot in that area.

1 → very easy to park
10 → extremely difficult

🧩 Visual Abstract

(Insert pipeline_diagram.png here)
A high-level overview of the detection, feature extraction, and regression components used to generate the difficulty score.

📚 Dataset

ParkScope uses a combination of real, synthetic, and auto-labeled data:

Real street images
From datasets like Cityscapes and BDD100K.

Synthetic images
Generated with Stable Diffusion to create scenes with varying densities, lighting, weather, and street layouts.

Auto-generated labels
Using parking density heuristics derived from object detection & curb segmentation.

Numerical features

car_count

curb_length

density_ratio

Additional contextual features extracted from segmentation.

🔧 Data Augmentation & Generation

To improve robustness and expand the dataset:

Synthetic variations: crowded vs. empty streets

Time-of-day changes: morning, afternoon, night

Weather simulation: rain, fog, low visibility

Image augmentations: brightness, contrast, blur, shadows

Labeling Pipeline:
YOLO/DETR (car detection) → curb segmentation → density calculation → difficulty score

🧠 Models & Pipelines
1. Detection Layer

YOLO/DETR for identifying parked vehicles and extracting bounding boxes.

2. Segmentation Layer

Curb and road segmentation to estimate available parking space.

3. Feature Computation

Density ratio = number_of_cars / curb_length

4. Regression Models

XGBoost Regressor (baseline for numeric features)

ResNet50 (fine-tuned vision regression model)

Vision Transformer (ViT-B/16) (state-of-the-art deep learning model)

🏋️ Training Process

Train XGBoost baseline using extracted features.

Fine-tune ResNet50 and ViT on full images.

Compare model performance across the same validation/test sets.

Tune hyperparameters and experiment with augmentation strategies.

Evaluate and refine density-based labeling heuristics.

📏 Metrics

Models are evaluated using:

MAE – Mean Absolute Error

RMSE – Root Mean Square Error

Difficulty Category Accuracy (easy / medium / hard)

Pearson Correlation between prediction and true density

📈 Results

(Insert plots, tables, and visualizations here.)
Examples include:

Comparison charts of model performance

Error distribution graphs

Qualitative examples of predicted scores vs. ground truth

📁 Repository Structure

(Insert the structure you approved earlier.)

👤 Team Members

Lior Yanwo
