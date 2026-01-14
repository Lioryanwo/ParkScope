import cv2
from ultralytics import YOLO

# 1. Loading the final model trained on the comprehensive dataset (v2).
MODEL_PATH = r'C:\ParkScope-main\ParkScope-main\runs\detect\parkscope_final_model\weights\best.pt'
model = YOLO(MODEL_PATH)

def analyze_parking_optimized(img_path):
    img = cv2.imread(img_path)
    if img is None: return

    # 2. Inference with Balanced Confidence (0.25) to mitigate sidewalk false positives
    # Utilizing 'device=0' for high-performance GPU inference (1.0ms)
    results = model(img_path, conf=0.25, device=0)[0]
    
    # Mapping classes from both original and synthetic datasets.
    parking_classes = ['available', 'parking_slot', 'parking_spot', 'slot']
    vehicle_classes = ['occupied', 'car', 'bus', 'truck', 'van']

    found_slots = []
    found_vehicles = []

    for box in results.boxes:
        b = box.xyxy[0].cpu().numpy()
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        
        # Parking space detection (Both original and synthetic)
        if cls_name in parking_classes:
            found_slots.append({'box': b, 'conf': conf, 'label': cls_name})
        # Detection of occupied parking spots.
        elif cls_name in vehicle_classes:
            found_vehicles.append({'box': b, 'conf': conf})

    # 3. Visualizing and Validating Results
    # Vehicles are always marked in red (BUSY status)
    for v in found_vehicles:
        cv2.rectangle(img, (int(v['box'][0]), int(v['box'][1])), 
                      (int(v['box'][2]), int(v['box'][3])), (0, 0, 255), 2)
        cv2.putText(img, f"BUSY ({v['conf']:.2f})", (int(v['box'][0]), int(v['box'][1]-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Available spots (including GenAI-generated) are marked in green (FREE status).
    for s in found_slots:
        cv2.rectangle(img, (int(s['box'][0]), int(s['box'][1])), 
                      (int(s['box'][2]), int(s['box'][3])), (0, 255, 0), 2)
        cv2.putText(img, f"FREE ({s['conf']:.2f})", (int(s['box'][0]), int(s['box'][1]-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite('parkscope_balanced_output.jpg', img)
    print(f"Analysis complete. Result saved as parkscope_balanced_output.jpg")

if __name__ == '__main__':
    test_img = r'C:\Parking_Spot_GenAI\Dataset KITTI\data_object_image_2\training\image_2\000010.png'
    analyze_parking_optimized(test_img)