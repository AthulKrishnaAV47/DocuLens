import cv2
import scipy
import numpy as np
import skimage
import pytesseract

class DocuLensEngine:
    def __init__(self):
        self.state_history = {}

    def access_quality(self, crop_image):
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
    
        
