import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os

# 1. טעינת המודלים
print("Loading Models...")
yolo_model = YOLO(r'C:\ParkScope-main\ParkScope-main\runs\detect\vehicle_detection_final\weights\best.pt')
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    variant="fp16",
    torch_dtype=torch.float16
).to("cuda")

def generate_parking_dataset(img_path, output_folder):
    img = Image.open(img_path).convert("RGB")
    original_w, original_h = img.size
    
    # זיהוי רכבים עם YOLO
    results = yolo_model(img_path, conf=0.5)[0]
    boxes = results.boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        print(f"No cars found in {img_path}")
        return

    for i, box in enumerate(boxes):
        # יצירת מסיכה על הרכב
        mask = np.zeros((original_h, original_w), dtype=np.uint8)
        x1, y1, x2, y2 = box
        # נרחיב קצת את המסיכה כדי למחוק גם צל
        mask[int(y1)-10 : int(y2)+10, int(x1)-10 : int(x2)+10] = 255
        mask_img = Image.fromarray(mask)

        # הרצת Inpainting
        print(f"Inpainting car {i+1}/{len(boxes)} in {os.path.basename(img_path)}...")
        prompt = "empty asphalt parking spot, grey street, urban curbside"
        synthetic_img = pipe(prompt=prompt, image=img, mask_image=mask_img).images[0]

        # שמירה
        save_path = os.path.join(output_folder, f"synthetic_{os.path.basename(img_path).split('.')[0]}_{i}.jpg")
        synthetic_img.save(save_path)

if __name__ == "__main__":
    input_dir = r'C:\Parking_Spot_GenAI\Dataset KITTI\data_object_image_2\training\image_2'
    output_dir = r'C:\ParkScope-main\ParkScope-main\data\synthetic_empty_slots'
    os.makedirs(output_dir, exist_ok=True)

    # הרצה על 10 התמונות הראשונות כדוגמה ל-Demo
    images = [f for f in os.listdir(input_dir) if f.endswith('.png')][:200]
    for img_name in images:
        generate_parking_dataset(os.path.join(input_dir, img_name), output_dir)