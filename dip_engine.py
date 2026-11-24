import cv2
import numpy as np
from detection import detect_corners, warp_from_corners


class DocuLensEngine:
    """Classical CPU-based document enhancement and quality diagnostic engine."""

    def analyze_quality(self, image: np.ndarray) -> dict:
        """Evaluates sharpness, contrast, and specular glare for UI warnings."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = "Green" if lap_var >= 150 else ("Yellow" if lap_var >= 60 else "Red")

        std_dev = float(np.std(gray))
        contrast = "Green" if std_dev >= 55 else ("Yellow" if std_dev >= 35 else "Red")

        glare_ratio = float(np.sum(gray >= 245)) / gray.size
        glare = "Green" if glare_ratio <= 0.05 else ("Yellow" if glare_ratio <= 0.15 else "Red")

        advice_parts = []
        if sharpness == "Red":
            advice_parts.append("Image is blurry; hold steady or tap to focus.")
        if contrast == "Red":
            advice_parts.append("Low contrast; improve ambient lighting.")
        if glare == "Red":
            advice_parts.append("Heavy glare detected; tilt camera slightly away from light.")
        advice = " ".join(advice_parts) if advice_parts else "Good capture quality."

        return {
            "flags": {"sharpness": sharpness, "contrast": contrast, "glare": glare},
            "raw": {
                "laplacian_var": round(lap_var, 2),
                "std_dev": round(std_dev, 2),
                "glare_ratio": round(glare_ratio, 4),
            },
            "advice": advice,
        }

    def _remove_shadows(self, gray: np.ndarray) -> np.ndarray:
        """Estimates uneven background lighting and normalizes it to pure white."""
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        norm = 255 - cv2.absdiff(gray, bg)
        return cv2.normalize(norm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    def apply_color_scan(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Preset 1 (Default): Photo-safe LAB enhancement.
        Brightens lighting and sharpens text while preserving faces, stamps, and colored ink.
        """
        if len(bgr_image.shape) == 2:
            return self.apply_bw_crisp(bgr_image)

        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)

        gaussian = cv2.GaussianBlur(l_eq, (0, 0), 1.5)
        l_sharp = cv2.addWeighted(l_eq, 1.35, gaussian, -0.35, 0)

        p2, p98 = np.percentile(l_sharp, (2, 98))
        if p98 > p2:
            l_final = np.clip(
                (l_sharp.astype(np.float32) - p2) * (255.0 / (p98 - p2)), 0, 255
            ).astype(np.uint8)
        else:
            l_final = l_sharp

        return cv2.cvtColor(cv2.merge([l_final, a, b]), cv2.COLOR_LAB2BGR)

    def apply_bw_crisp(self, image: np.ndarray) -> np.ndarray:
        """
        Preset 2: Shadow-normalized black & white scan.
        Best for receipts, printed forms, and handwritten notes.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        norm = self._remove_shadows(gray)
        blurred = cv2.GaussianBlur(norm, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.medianBlur(thresh, 3)

    def enhance(self, image: np.ndarray, preset: str = "color_scan") -> np.ndarray:
        """
        Applies the selected enhancement preset.
        preset: 'color_scan' | 'bw_crisp' | 'original'
        """
        if preset == "bw_crisp":
            return self.apply_bw_crisp(image)
        if preset == "original":
            return image.copy()
        return self.apply_color_scan(image)


_engine = DocuLensEngine()


def process_scan(
    image: np.ndarray,
    corners: list | None = None,
    preset: str = "color_scan",
    crop: bool = True,
) -> dict:
    """
    Production entry point for backend/frontend integration.
    - crop=True + corners provided: warps using user-adjusted corners.
    - crop=True + corners=None: auto-detects (falls back to full image if missed).
    - crop=False: enhances the full uncropped image.
    """
    target = image
    was_cropped = False

    if crop:
        if corners is not None:
            target = warp_from_corners(image, corners)
            was_cropped = True
        else:
            auto_corners, detected = detect_corners(image)
            if detected:
                target = warp_from_corners(image, auto_corners)
                was_cropped = True

    enhanced = _engine.enhance(target, preset=preset)
    diagnostics = _engine.analyze_quality(target)

    return {
        "enhanced_image": enhanced,
        "cropped": was_cropped,
        "preset": preset,
        "diagnostics": diagnostics,
    }