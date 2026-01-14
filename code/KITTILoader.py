import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob

KITTI_IMG_PATH = r"C:\Parking_Spot_GenAI\Dataset KITTI\data_object_image_2\training\image_2\*.png" 
OUTPUT_IMG_DIR = r"C:\Parking_Spot_GenAI\synthetic_data\images"
OUTPUT_LBL_DIR = r"C:\Parking_Spot_GenAI\synthetic_data\labels"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

vehicle_detector = YOLO('yolo11m.pt') 

def create_high_quality_synthetic():
    images = glob.glob(KITTI_IMG_PATH)
    count = 0
    max_images = 500 # limit test to 500 photos

    for img_path in images:
        if count >= max_images:
            break
            
        img = cv2.imread(img_path)
        results = vehicle_detector(img, half=True, device=0) # to speed thinks up we used GPU.
        
        # Vehicle Filtering: Only vehicles with a width exceeding 100 pixels (to ensure clarity).
        boxes = []
        for box in results[0].boxes:
            if int(box.cls) == 2: # car
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                if (x2 - x1) > 100: # Filtering Small/Distant Vehicles.
                    boxes.append([x1, y1, x2, y2])
        
        if len(boxes) >= 2:
            target = boxes[np.random.randint(0, len(boxes))]
            x1, y1, x2, y2 = target
            
            # Intelligent Object Removal (Inpaint).
            mask = np.zeros(img.shape[:2], np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            img_synthetic = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            
            # Asphalt-based Texture Filling (Noise Injection).
            roi = img_synthetic[y1:y2, x1:x2]
            noise = np.random.normal(0, 5, roi.shape).astype(np.uint8)
            img_synthetic[y1:y2, x1:x2] = cv2.add(roi, noise)
            
            # save to..
            fname = os.path.basename(img_path)
            cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, fname), img_synthetic)
            
            # Lableing 
            h, w, _ = img.shape
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            
            with open(os.path.join(OUTPUT_LBL_DIR, fname.replace('.png', '.txt')), 'w') as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            
            count += 1
            print(f"✅ [{count}/{max_images}] Successfully synthesized high-quality parking space for: {fname}")

if __name__ == "__main__":
    create_high_quality_synthetic()