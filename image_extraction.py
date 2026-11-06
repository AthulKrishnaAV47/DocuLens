from datasets import load_dataset
import os
import json

dataset = load_dataset("mapo80/DocCornerDataset", split="train", streaming=True)

os.makedirs("doculens_dataset/images", exist_ok=True)
ground_truth = {}

count = 0
for sample in dataset:
    if count >= 150:
        break
        
    if sample.get('is_negative', False):
        continue
    
    image = sample['image']
    
    corners = [
        [sample['corner_tl_x'], sample['corner_tl_y']], # Top-Left
        [sample['corner_tr_x'], sample['corner_tr_y']], # Top-Right
        [sample['corner_br_x'], sample['corner_br_y']], # Bottom-Right
        [sample['corner_bl_x'], sample['corner_bl_y']]  # Bottom-Left
    ]
    
    filename = f"doc_{count}.jpg"
    filepath = os.path.join("doculens_dataset/images", filename)
    
    image.save(filepath)
    
    ground_truth[filename] = corners
    count += 1

with open("doculens_dataset/ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=4)

print(f"Successfully downloaded {count} images and generated ground_truth.json!")