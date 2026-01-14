# 🚗 ParkScope: GenAI-Enhanced Street Parking Detection

ParkScope is a sophisticated computer vision system designed to detect **available** and **occupied** street parking spots in real-time. By leveraging **YOLO11** and bridging the data scarcity gap with **Generative AI (Stable Diffusion)**, ParkScope achieves high reliability in dynamic urban environments.

## 🧠 Core Methodology & Novelty

The project addresses the "Empty Spot" data deficiency by implementing a **Synthetic Data Pipeline**:

* **GenAI Inpainting:** Used Stable Diffusion to remove vehicles from KITTI dataset images, creating realistic "available" parking spot samples.
* **Hybrid Decision Logic:** The system cross-references direct "available" detections with "vehicle" detections to ensure maximum reliability and prevent false positives.
* **Spatial Awareness:** Trained with advanced augmentations including **Perspective Transformation** and **Mosaic** to handle complex camera angles and varying distances.

## 📈 Performance & Results

* **High Accuracy:** Achieved a **mAP50 of 88.8%** for available parking spot detection.
* **Real-Time Inference:** Optimized for high-speed processing, reaching **~1.0ms** inference time on NVIDIA RTX 4070.
* **Robustness:** Successfully generalizes across different street layouts and lighting conditions.

*(Example: Green boxes = FREE | Red boxes = BUSY)*

---

## 📂 Repository Structure (Polished)

* **`code/`**: Core implementation scripts (Download, GenAI Data Gen, Training, Inference).
* **`data/synthetic_empty_slots/`**: The proprietary dataset of AI-generated empty parking spots.
* **`runs/detect/parkscope_final_model/`**: Production-ready model weights (`best.pt`).
* **`slides/`**: Project presentations and interim reports in PDF format.

---

## 🚀 Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Lioryanwo/ParkScope.git
cd ParkScope

# Install required dependencies
pip install -r requirements.txt

```

### 2. Run the System

To analyze a parking scene using the GenAI-enhanced model:

```bash
python code/final_parking_system.py

```

---

## 🛠 Tech Stack

* **Architecture:** YOLO11 (Ultralytics)
* **Generative AI:** Stable Diffusion (Inpainting)
* **Computer Vision:** OpenCV, NumPy
* **Dataset:** KITTI Vision Benchmark + Custom Synthetic Data

---

**זהו השלב האחרון!** עכשיו, כשקובץ ה-README שלך נראה מקצועי ומסכם את כל העבודה הקשה, הפרויקט מוכן להגשה רשמית.

**האם תרצה שאכין לך "דף מסרים" קצר (Cheat Sheet) למצגת הסופית, עם 3 הנקודות הכי חשובות שאתה חייב להדגיש מול המרצה?**