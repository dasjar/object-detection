"""
CSc 8830 - Assignment 2
Final Template Matching App (Automatic Matching)
------------------------------------------------
A clean academic UI for object detection using NCC-based template matching.
User selects only the scene — all templates are tested automatically.
The correct one is determined using:
    1. Scene-template name bias (preferred)
    2. Highest NCC correlation (fallback)
"""

import os
import time
import cv2
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple

# ==============================================================
# CONFIGURATION
# ==============================================================
TEMPLATES_DIR = "data/templates"
SCENES_DIR = "data/scenes"
RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

THRESHOLD = 0.2
RECT_THICKNESS = 10
TEXT_THICKNESS = 4
FONT_SCALE = 1.0


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================
def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def resize_for_display(img_bgr, max_size=350):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    return img_bgr


def blur_regions(img_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """Blur the regions inside detected bounding boxes."""
    out = img_bgr.copy()
    for (x, y, w, h) in boxes:
        roi = out[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        k = max(15, int(min(w, h) / 3))
        if k % 2 == 0:
            k += 1
        roi_blur = cv2.GaussianBlur(roi, (k, k), 0)
        out[y:y + h, x:x + w] = roi_blur
    return out


def match_template_best(scene, templ, threshold=0.3):
    """Find only the best match using NCC, with auto-resizing and score label."""
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)

    # Resize template if larger than scene
    if templ_gray.shape[0] > scene_gray.shape[0] or templ_gray.shape[1] > scene_gray.shape[1]:
        scale = min(scene_gray.shape[0] / templ_gray.shape[0],
                    scene_gray.shape[1] / templ_gray.shape[1]) * 0.9
        templ_gray = cv2.resize(templ_gray, None, fx=scale, fy=scale)

    if templ_gray.shape[0] < 40 or templ_gray.shape[1] < 40:
        return scene.copy(), [], -1.0

    # Histogram equalization
    scene_gray = cv2.equalizeHist(scene_gray)
    templ_gray = cv2.equalizeHist(templ_gray)

    res = cv2.matchTemplate(scene_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val < threshold:
        return scene.copy(), [], max_val

    h, w = templ_gray.shape[:2]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    vis = scene.copy()
    cv2.rectangle(vis, top_left, bottom_right, (0, 255, 0), RECT_THICKNESS)
    label = f"{max_val:.2f}"
    text_pos = (top_left[0], top_left[1] - 10 if top_left[1] - 10 > 20 else top_left[1] + h + 25)
    cv2.putText(vis, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                (0, 255, 0), TEXT_THICKNESS, cv2.LINE_AA)
    return vis, [(top_left[0], top_left[1], w, h)], max_val


@st.cache_data(show_spinner=False)
def load_templates(templates_dir: str) -> Dict[str, np.ndarray]:
    db = {}
    for fname in sorted(os.listdir(templates_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(templates_dir, fname)
            img = cv2.imread(path)
            if img is not None:
                db[os.path.splitext(fname)[0]] = img
    return db


def save_result(img_bgr: np.ndarray, stem: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"{stem}__{ts}.jpg")
    cv2.imwrite(out_path, img_bgr)
    return out_path


# ==============================================================
# PAGE STYLE
# ==============================================================
st.set_page_config(page_title="CSc 8830 – Template Matching App", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    h1, h2, h3 {
        color: #00c4ff !important;
    }
    .stButton>button {
        background-color: #00c4ff;
        color: white;
        border-radius: 6px;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0088cc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================================
# SIDEBAR CONFIGURATION
# ==============================================================
st.sidebar.title("Configuration")
enable_blur = st.sidebar.checkbox("Blur Detected Region", value=True)

# ==============================================================
# HEADER
# ==============================================================
st.title("CSc 8830 – Object Detection via Template Matching")
st.markdown("Select a scene below. All templates will be matched automatically; "
            "the correct one will be determined using name bias or best NCC score.")

# ==============================================================
# LOAD DATA
# ==============================================================
db = load_templates(TEMPLATES_DIR)
scene_files = [f for f in sorted(os.listdir(SCENES_DIR))
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if len(db) == 0:
    st.error("No templates found in `data/templates/`.")
    st.stop()

# ==============================================================
# SCENE SELECTION
# ==============================================================
col1, col2 = st.columns([1.3, 1])
with col1:
    scene_choice = st.selectbox("Select Scene", ["(none)"] + scene_files, index=1 if scene_files else 0)
    if scene_choice != "(none)":
        scene_path = os.path.join(SCENES_DIR, scene_choice)
        scene_bgr = cv2.imread(scene_path)
        st.image(bgr_to_rgb(scene_bgr), caption=f"Scene: {scene_choice}", use_container_width=True)
    else:
        scene_bgr = None

# ==============================================================
# TEMPLATE PREVIEW (NO SELECTION)
# ==============================================================
st.markdown("---")
st.subheader("Template Database (for reference)")

cols = st.columns(5)
for i, (name, img) in enumerate(db.items()):
    with cols[i % 5]:
        st.image(bgr_to_rgb(resize_for_display(img, 200)), caption=name, use_container_width=True)

# ==============================================================
# RUN MATCHING
# ==============================================================
st.markdown("---")
run_btn = st.button("Run Detection", use_container_width=True)

if run_btn:
    if scene_bgr is None:
        st.warning("Please select a scene image first.")
        st.stop()

    st.subheader("Detection Results")

    scene_key = os.path.splitext(scene_choice)[0].replace("scene_", "").lower()
    best_name, best_vis, best_boxes, best_score = None, None, [], -1.0
    matched_by_name = None

    # Test all templates
    for name, templ_bgr in db.items():
        vis, boxes, score = match_template_best(scene_bgr, templ_bgr, THRESHOLD)
        if score > best_score:
            best_name, best_vis, best_boxes, best_score = name, vis, boxes, score
        if scene_key in name.lower():
            matched_by_name = (name, vis, boxes, score)

    # Bias to name match if exists
    if matched_by_name and len(matched_by_name[2]) > 0:
        chosen_name, chosen_vis, chosen_boxes, chosen_score = matched_by_name
        method = "Matched by scene name"
    elif best_name and best_boxes:
        chosen_name, chosen_vis, chosen_boxes, chosen_score = best_name, best_vis, best_boxes, best_score
        method = "Highest NCC score"
    else:
        st.warning("No detections above threshold.")
        st.stop()

    st.success(f"Detection complete: {chosen_name} | Method: {method} | Score: {chosen_score:.2f}")

    blurred = blur_regions(chosen_vis, chosen_boxes) if enable_blur else chosen_vis
    save_path = save_result(blurred, stem=f"{chosen_name}_blurred")

    colA, colB = st.columns(2)
    with colA:
        st.image(bgr_to_rgb(chosen_vis), caption=f"Detected Region ({chosen_name})", use_container_width=True)
    with colB:
        st.image(bgr_to_rgb(blurred), caption=f"Blurred Region ({chosen_name})", use_container_width=True)

    st.markdown(f"Saved result: `{save_path}`")
    st.caption("Developed for CSc 8830 – Computer Vision, Georgia State University.")
