import time
import cv2
import numpy as np
import streamlit as st
from detection import detect_corners
from dip_engine import process_scan

st.set_page_config(page_title="DocuLens Engine Tester", layout="wide")
st.title("DocuLens — Engine Test UI")


def draw_crop_overlay(image: np.ndarray, corners: list) -> np.ndarray:
    """Draws the 4-point crop polygon and labeled corner handles on the image."""
    overlay = image.copy()
    pts = np.array(corners, dtype=np.int32)
    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

    labels = ["1: TL", "2: TR", "3: BR", "4: BL"]
    for (x, y), label in zip(pts, labels):
        cv2.circle(overlay, (int(x), int(y)), 10, (0, 0, 255), -1)
        cv2.putText(
            overlay, label, (int(x) + 12, int(y) + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
    return overlay


preset = st.selectbox("Enhancement Preset", ["color_scan", "bw_crisp", "original"], index=0)
uploaded_file = st.file_uploader("Upload a document photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if original_bgr is None:
        st.error("Could not decode the uploaded image.")
        st.stop()

    h, w = original_bgr.shape[:2]

    # Step 1: Run Auto-Crop detection
    auto_corners, auto_detected = detect_corners(original_bgr)

    # Step 2: Decide between Auto-Crop, Manual Fallback, or No Crop
    use_crop = True
    manual_mode = False

    if auto_detected:
        st.success("Auto-crop detected the document boundaries.")
        crop_choice = st.radio(
            "Do you want to keep the auto-crop or manually choose the points?",
            ["Keep Auto-Crop", "Manually Choose Points", "No Crop (Full Image)"],
            horizontal=True
        )
        if crop_choice == "Manually Choose Points":
            manual_mode = True
        elif crop_choice == "No Crop (Full Image)":
            use_crop = False
    else:
        st.warning("Auto-crop could not find clean edges. Please manually choose the 4 corner points below (or skip cropping).")
        crop_choice = st.radio(
            "Fallback Option",
            ["Manually Choose Points", "No Crop (Full Image)"],
            horizontal=True
        )
        if crop_choice == "Manually Choose Points":
            manual_mode = True
        else:
            use_crop = False

    # Step 3: Manual 4-point corner controls (pre-filled with auto or 5% inset corners)
    active_corners = auto_corners
    if manual_mode:
        st.markdown("#### Adjust Corner Points (X, Y)")
        c1, c2, c3, c4 = st.columns(4)
        corner_names = ["1: Top-Left (TL)", "2: Top-Right (TR)", "3: Bottom-Right (BR)", "4: Bottom-Left (BL)"]
        cols = [c1, c2, c3, c4]

        manual_corners = []
        for idx, (col, name) in enumerate(zip(cols, corner_names)):
            with col:
                st.caption(name)
                x_val = st.slider(f"X{idx+1}", 0, w - 1, int(auto_corners[idx][0]), key=f"x_{idx}")
                y_val = st.slider(f"Y{idx+1}", 0, h - 1, int(auto_corners[idx][1]), key=f"y_{idx}")
                manual_corners.append([float(x_val), float(y_val)])
        active_corners = manual_corners

    # Step 4: Warp + Enhance
    t0 = time.perf_counter()
    out = process_scan(
        original_bgr,
        corners=active_corners if use_crop else None,
        preset=preset,
        crop=use_crop
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    enhanced = out["enhanced_image"]
    diag = out["diagnostics"]

    mode_label = "Manual Crop" if manual_mode else ("Auto-Crop" if use_crop else "Uncropped")
    st.caption(
        f"**Mode:** {mode_label} | "
        f"**Preset:** `{out['preset']}` | "
        f"**Quality:** {diag['advice']} | "
        f"**Time:** {elapsed_ms:.0f} ms"
    )

    # Side-by-side preview (shows live crop box on Original when cropping is enabled)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original" + (" (Crop Box)" if use_crop else ""))
        preview_orig = draw_crop_overlay(original_bgr, active_corners) if use_crop else original_bgr
        st.image(cv2.cvtColor(preview_orig, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("Enhanced Output")
        display_img = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB) if len(enhanced.shape) == 3 else enhanced
        st.image(display_img, use_container_width=True)

    success, png_buffer = cv2.imencode(".png", enhanced)
    if success:
        st.download_button(
            label="Download Enhanced PNG",
            data=png_buffer.tobytes(),
            file_name=f"doculens_{out['preset']}.png",
            mime="image/png"
        )