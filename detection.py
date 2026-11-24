import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 corner points consistently as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _is_valid_quad(approx: np.ndarray, min_ratio: float = 0.25, max_ratio: float = 4.0) -> bool:
    """Rejects extreme strips (pens, rulers) and heavily skewed trapezoids."""
    pts = order_points(approx.reshape(4, 2))
    (tl, tr, br, bl) = pts

    w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    if h == 0 or w == 0 or not (min_ratio <= (w / h) <= max_ratio):
        return False

    ordered = [tl, tr, br, bl]
    for i in range(4):
        v1 = ordered[(i - 1) % 4] - ordered[i]
        v2 = ordered[(i + 1) % 4] - ordered[i]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            return False
        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)))
        if not (65.0 <= angle <= 115.0):
            return False

    return True


def _find_quad(binary_map: np.ndarray, img_area: float, min_area_ratio: float = 0.10):
    """Finds the largest valid convex 4-point contour within the area bounds."""
    contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_ratio * img_area or area > 0.98 * img_area:
            continue

        peri = cv2.arcLength(c, True)
        for eps in np.linspace(0.01, 0.05, 9):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx) and _is_valid_quad(approx):
                return approx

    return None


def _default_inset_corners(h: int, w: int, margin: float = 0.05) -> list:
    """Returns a 5% inset rectangle so the frontend always has draggable corner handles."""
    mx, my = w * margin, h * margin
    return [
        [round(mx, 1), round(my, 1)],
        [round(w - mx, 1), round(my, 1)],
        [round(w - mx, 1), round(h - my, 1)],
        [round(mx, 1), round(h - my, 1)],
    ]


def detect_corners(image_input) -> tuple[list, bool]:
    """
    Detects document corners using a 3-pass classical CV pipeline:
      Pass 1: Text suppression + Adaptive Auto-Canny
      Pass 2: Otsu brightness segmentation
      Pass 3: HSV saturation mask

    Accepts an in-memory BGR np.ndarray or a file path string.
    Returns: (corners, auto_detected)
      - corners: JSON-ready list of 4 [x, y] points (TL, TR, BR, BL)
      - auto_detected: True if detected, False if fallback inset corners were returned
    """
    image = cv2.imread(image_input) if isinstance(image_input, str) else image_input
    if image is None:
        return [], False

    orig_h, orig_w = image.shape[:2]
    target_h = 800.0
    ratio = orig_h / target_h
    resized = cv2.resize(image, (int(orig_w / ratio), int(target_h)))
    img_area = resized.shape[0] * resized.shape[1]

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # PASS 1: Text suppression + Adaptive Auto-Canny
    text_removed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, morph_kernel)
    blurred = cv2.GaussianBlur(text_removed, (5, 5), 0)
    v = np.median(blurred)
    edged = cv2.Canny(blurred, int(max(0, 0.67 * v)), int(min(255, 1.33 * v)))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, edge_kernel)
    screen_cnt = _find_quad(edged, img_area)

    # PASS 2: Otsu brightness segmentation
    if screen_cnt is None:
        _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, morph_kernel)
        screen_cnt = _find_quad(otsu_mask, img_area)

    # PASS 3: HSV saturation mask (isolates low-saturation paper from colored/wood desks)
    if screen_cnt is None:
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
        _, sat_mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, morph_kernel)
        screen_cnt = _find_quad(sat_mask, img_area)

    if screen_cnt is None:
        return _default_inset_corners(orig_h, orig_w), False

    pts = order_points(screen_cnt.reshape(4, 2) * ratio)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in pts], True


def warp_from_corners(image: np.ndarray, corners) -> np.ndarray:
    """Applies a perspective transform using 4 [x, y] corner coordinates."""
    pts = np.array(corners, dtype="float32").reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    max_w = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)), 1)
    max_h = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)), 1)

    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_w, max_h))


def detect_and_warp(image_input):
    """Convenience wrapper for batch scripts like test_pipeline.py."""
    image = cv2.imread(image_input) if isinstance(image_input, str) else image_input
    if image is None:
        return None, None
    corners, detected = detect_corners(image)
    if not detected:
        return image, None
    return warp_from_corners(image, corners), np.array(corners, dtype="float32")