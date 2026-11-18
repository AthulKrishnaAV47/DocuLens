import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pytesseract
from pytesseract import Output

class DocuLensEngine:
    def __init__(self):
        self.state_history = {}

    def assess_quality(self, crop_image):
        """Analyze the raw image and output multi-dimensional quality vector"""

        if len(crop_image) == 3:
            gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop_image

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_flag = "Red" if laplacian_var < 100 else ("Yellow" if laplacian_var < 300 else "Green")

        contrast_std = np.std(gray)
        contrast_flag = "Red" if contrast_std < 30 else ("Yellow" if contrast_std < 50 else "Green")

        glare_mask = cv2.inRange(gray, 240, 255)
        glare_ratio = np.sum(glare_mask > 0) / gray.size
        glare_flag = "Red" if glare_ratio > 0.05 else ("Yellow" if glare_ratio > 0.02 else "Green")

        advice = []
        if sharpness_flag == "Red": advice.append("Image is heavily blurred. Hold the camera steady.")
        if contrast_flag == "Red": advice.append("Poor contrast. Move to a well-lit environment.")
        if glare_flag == "Red": advice.append("Severe glare detected. Move away from direct overhead light.")
        if not advice: advice.append("Lighting and focus look good!")

        diagnostics = {
            "metrics": {
                "sharpness": round(laplacian_var, 2),
                "contrast": round(contrast_std, 2),
                "glare_ratio": round(glare_ratio, 4)
            },
            "flags": {
                "sharpness": sharpness_flag,
                "contrast": contrast_flag,
                "glare": glare_flag
            },
            "advice": " ".join(advice)
        }
        self.state_history["diagnostics"] = diagnostics
        return diagnostics

    def apply_standard_otsu(self, gray):
            """Method: Standard Global Thresholding"""
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.state_history["Standard_otsu"] = thresh
            return thresh
    def apply_adaptive_threshold(self, gray):
                """Method: Localized Adaptive Thresholding for shadows/glare"""
                filtered = cv2.bilateralFilter(gray, 9, 75, 75) 
                thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
                self.state_history["adaptive_thresh"] = thresh
                return thresh

    def apply_unsharp_mask(self, gray):
        """Method C: Sharpening kernel prior to thresholding for blurred images[cite: 6]."""
        gaussian = cv2.GaussianBlur(gray, (9, 9), 10.0)
        sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.state_history["unsharp_mask"] = thresh
        return thresh


    def calculate_metrics(self, original_gray, processed_image):
        """Calculates SSIM, PSNR, and OCR confidence scores[cite: 3, 6]."""
        # Determine the smallest dimension of the image
        min_dim = min(original_gray.shape)
        
        # Dynamically handle the SSIM win_size constraint
        if min_dim < 7:
            if min_dim < 3:
                ssim_score = 0.0  # Image is too small for any structural comparison
            else:
                # win_size must be an odd number less than or equal to min_dim
                win_size = min_dim if min_dim % 2 != 0 else min_dim - 1
                ssim_score = ssim(original_gray, processed_image, win_size=win_size)
        else:
            ssim_score = ssim(original_gray, processed_image) # Uses default win_size=7

        psnr_score = psnr(original_gray, processed_image)
        
        # Pytesseract OCR evaluation[cite: 3]
        ocr_data = pytesseract.image_to_data(processed_image, output_type=Output.DICT)
        confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) != -1]
        avg_ocr_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "SSIM": round(ssim_score, 4),
            "PSNR": round(psnr_score, 2),
            "OCR_Confidence": round(avg_ocr_conf, 2)
        }

    def select_optimal_pipeline(self, diagnostics):
        """Lightweight decision module to automatically select the optimal pipeline[cite: 6]."""
        flags = diagnostics["flags"]
        
        if flags["glare"] == "Red" or flags["contrast"] == "Red":
            return "Adaptive"
        elif flags["sharpness"] == "Red":
            return "Sharpening"
        else:
            return "Standard"


    def process_document(self, cropped_bgr):
        """The main API endpoint that Sridhar C's frontend will call[cite: 9]."""
        self.state_history = {} # Reset state
        self.state_history["original_crop"] = cropped_bgr
        
        gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
        self.state_history["grayscale"] = gray

        diagnostics = self.assess_quality(gray)

        otsu_res = self.apply_standard_otsu(gray)
        adaptive_res = self.apply_adaptive_threshold(gray)
        sharpen_res = self.apply_unsharp_mask(gray)

        metrics = {
            "Standard": self.calculate_metrics(gray, otsu_res),
            "Adaptive": self.calculate_metrics(gray, adaptive_res),
            "Sharpening": self.calculate_metrics(gray, sharpen_res)
        }

        best_method = self.select_optimal_pipeline(diagnostics)

        payload = {
            "diagnostics": diagnostics,
            "recommended_method": best_method,
            "comparative_metrics": metrics,
            "intermediate_images": self.state_history
        }
        return payload
