import cv2
import json
import os
from dip_engine import DocuLensEngine
from detection import detect_and_warp 

def run_evaluation():
    engine = DocuLensEngine()
    dataset_dir = "doculens_dataset/images"
    
    with open("doculens_dataset/ground_truth.json", "r") as f:
        ground_truth = json.load(f)

    total_images = len(ground_truth)
    detection_successes = 0
    failure_log = []

    print(f"Starting evaluation on {total_images} images...")

    for filename, true_corners in ground_truth.items():
        filepath = os.path.join(dataset_dir, filename)
        
        warped_crop, detected_corners = detect_and_warp(filepath)
        
        if detected_corners is not None:
            detection_successes += 1
            
            results = engine.process_document(warped_crop)
            
            print(f"\n[PASS] {filename}")
            print(f"Quality Flags: {results['diagnostics']['flags']}")
            print(f"Auto-Selected Pipeline: {results['recommended_method']}")
            print(f"Standard OCR: {results['comparative_metrics']['Standard']['OCR_Confidence']}")
            print(f"Adaptive OCR: {results['comparative_metrics']['Adaptive']['OCR_Confidence']}")
            print(f"Sharpening OCR: {results['comparative_metrics']['Sharpening']['OCR_Confidence']}")
        else:
            print(f"\n[FAIL] {filename} - Boundary detection failed.")
            failure_log.append(filename)

    accuracy = (detection_successes / total_images) * 100
    print(f"\n=== FINAL METRICS ===")
    print(f"Detection Accuracy: {accuracy:.2f}% (Target: >= 60%)")
    print(f"Total Failures: {len(failure_log)}")

if __name__ == "__main__":
    run_evaluation()