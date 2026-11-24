import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _find_quad(binary_map, img_area, min_area_ratio=0.10):
    contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        area = cv2.contourArea(c)
        # Ignore tiny speckles AND contours that are basically the entire image border
        if area < min_area_ratio * img_area or area > 0.98 * img_area:
            continue

        peri = cv2.arcLength(c, True)
        for eps in np.linspace(0.01, 0.08, 10):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx
    return None

def detect_and_warp(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    original = image.copy()
    orig_h, orig_w = original.shape[:2]

    target_h = 800.0
    ratio = orig_h / target_h
    resized = cv2.resize(original, (int(orig_w / ratio), int(target_h)))
    img_area = resized.shape[0] * resized.shape[1]

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # PASS 1: Gentle text suppression + Auto-Canny
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    text_removed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, morph_kernel)
    blurred = cv2.GaussianBlur(text_removed, (5, 5), 0)

    v = np.median(blurred)
    lower = int(max(0, 0.67 * v))
    upper = int(min(255, 1.33 * v))
    edged = cv2.Canny(blurred, lower, upper)
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    screen_cnt = _find_quad(edged, img_area)

    # PASS 2 (Fallback): Otsu brightness segmentation for weak/soft edges
    if screen_cnt is None:
        _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, morph_kernel)
        screen_cnt = _find_quad(otsu_mask, img_area)

    if screen_cnt is None:
        return original, None

    pts = screen_cnt.reshape(4, 2) * ratio
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(original, M, (maxWidth, maxHeight))

    return warped, pts