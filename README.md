ParkScope – AI-Based Parking Difficulty Prediction
Project Motivation

Finding street parking in dense urban areas often leads to long search times, traffic congestion, and frustration.
Current navigation systems (Waze, Google Maps) do not provide any prediction of parking difficulty near the destination.

Problem Statement

Given a single street-level image, predict a Parking Difficulty Score (1–10) representing how hard it will be to find parking in that area.

Visual Abstract

(Add pipeline_diagram.png here)

Dataset

Real street images (Cityscapes, BDD100K)

Synthetic images generated with Stable Diffusion

Automatically labeled using density-based heuristics

Numerical features: car_count, curb_length, density_ratio

Data Augmentation & Generation

Synthetic street scenes (crowded / empty / different times of day)

Augmentations: brightness, shadows, blur, rain, night scenes

Auto-labeling: YOLO/DETR detection + curb segmentation → density → score

Models & Pipelines

1. YOLO/DETR – detect parked vehicles
2. Segmentation model – extract curb area
3. Feature computation – density ratio
4. Regression Models:

XGBoost Regressor

ResNet50 (fine-tuned regression)

Vision Transformer (ViT-B/16)

Training Process

Train baseline XGBoost on numeric features

Fine-tune ResNet & ViT on images

Compare all models across same test set

Hyperparameter tuning + augmentation refinements

Metrics

MAE

RMSE

Difficulty category accuracy

Pearson correlation

Results

(Add graphs from results/visualizations)

Repository Structure

(Use the tree structure above)

Team Members

Lior Yanwo
