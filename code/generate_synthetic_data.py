import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os

# 1. Model Loading:
print("Loading Models...")
# Updated to a relative path based on the project repository structure
YOLO_MODEL_PATH = os.path.join('runs', 'detect', 'vehicle_detection_final', 'weights', 'best.pt')
yolo_model = YOLO(YOLO_MODEL_PATH)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    variant="fp16",
    torch_dtype=torch.float16
).to("cuda")

def generate_parking_dataset(img_path, output_folder):
    img = Image.open(img_path).convert("RGB")
    original_w, original_h = img.size
    
    # Vehicle Detection using YOLO.
    results = yolo_model(img_path, conf=0.5)[0]
    boxes = results.boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        print(f"No cars found in {img_path}")
        return

    for i, box in enumerate(boxes):
        # Generating a Vehicle Mask.
        mask = np.zeros((original_h, original_w), dtype=np.uint8)
        x1, y1, x2, y2 = box
        # Expanding the mask area to ensure complete shadow removal.
        mask[max(0, int(y1)-10) : min(original_h, int(y2)+10), 
             max(0, int(x1)-10) : min(original_w, int(x2)+10)] = 255
        mask_img = Image.fromarray(mask)

        # Executing Image Inpainting.
        print(f"Inpainting car {i+1}/{len(boxes)} in {os.path.basename(img_path)}...")
        prompt = "empty asphalt parking spot, grey street, urban curbside"
        synthetic_img = pipe(prompt=prompt, image=img, mask_image=mask_img).images[0]

        # save to..
        # Using relative output folder path
        save_path = os.path.join(output_folder, f"synthetic_{os.path.basename(img_path).split('.')[0]}_{i}.jpg")
        synthetic_img.save(save_path)

if __name__ == "__main__":
    # Updated to relative paths consistent with the GitHub repository structure
    input_dir = os.path.join('Dataset KITTI', 'data_object_image_2', 'training', 'image_2')
    output_dir = os.path.join('data', 'synthetic_empty_slots')
    
    # Ensuring the output directory exists to prevent crashes
    os.makedirs(output_dir, exist_ok=True)

    """ Running a demo on the first 10 sample images
    images = [f for f in os.listdir(input_dir) if f.endswith('.png')][:10]
    for img_name in images:
        generate_parking_dataset(os.path.join(input_dir, img_name), output_dir)"""

    # Generating a comprehensive synthetic dataset from 200 KITTI samples
    if os.path.exists(input_dir):
        images = [f for f in os.listdir(input_dir) if f.endswith('.png')][:200]
        for img_name in images:
            generate_parking_dataset(os.path.join(input_dir, img_name), output_dir)
    else:
        print(f"Error: Input directory {input_dir} not found. Please check repository structure.")