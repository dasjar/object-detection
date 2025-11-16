"""
CSc 8830 - Assignment 2
Refined Research-Grade UI for Final Template Matching App
---------------------------------------------------------
This version preserves 100% of your functional logic,
but elevates the UI/UX to a professionally styled
academic interface suitable for a PhD-level project.
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
TEXT_THICKNESS = 5
FONT_SCALE = 2.0


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================
def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def resize_for_display(img_bgr, max_size=260):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    return img_bgr

def blur_regions(img_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    out = img_bgr.copy()
    for (x, y, w, h) in boxes:
        roi = out[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        k = max(15, int(min(w, h) / 3))
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur(roi, (k, k), 0)
        out[y:y + h, x:x + w] = blur
    return out

def match_template_best(scene, templ, threshold=0.3):
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)

    # Safe template resizing
    if templ_gray.shape[0] > scene_gray.shape[0] or templ_gray.shape[1] > scene_gray.shape[1]:
        scale = min(scene_gray.shape[0] / templ_gray.shape[0],
                    scene_gray.shape[1] / templ_gray.shape[1]) * 0.9
        templ_gray = cv2.resize(templ_gray, None, fx=scale, fy=scale)

    if templ_gray.shape[0] < 40 or templ_gray.shape[1] < 40:
        return scene.copy(), [], -1.0

    scene_gray = cv2.equalizeHist(scene_gray)
    templ_gray = cv2.equalizeHist(templ_gray)

    res = cv2.matchTemplate(scene_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val < threshold:
        return scene.copy(), [], max_val

    h, w = templ_gray.shape[:2]
    x, y = max_loc
    vis = scene.copy()

    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), RECT_THICKNESS)

    label = f"{max_val:.2f}"
    text_y = y - 10 if y - 10 > 20 else y + h + 25
    cv2.putText(vis, label, (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                (0, 255, 0), TEXT_THICKNESS, cv2.LINE_AA)

    return vis, [(x, y, w, h)], max_val

@st.cache_data(show_spinner=False)
def load_templates(templates_dir: str) -> Dict[str, np.ndarray]:
    db = {}
    for fname in sorted(os.listdir(templates_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imread(os.path.join(templates_dir, fname))
            if img is not None:
                db[os.path.splitext(fname)[0]] = img
    return db

def save_result(img_bgr: np.ndarray, stem: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"{stem}__{ts}.jpg")
    cv2.imwrite(out_path, img_bgr)
    return out_path


# ==============================================================
# PAGE STYLE — PROFESSIONAL DARK UI
# ==============================================================
st.set_page_config(page_title="Template Matching App", layout="wide")

st.markdown(
    """
    <style>
    /* Layout */
    .stApp { background-color: #0c0e12; color: #e6e6e6; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* Headings */
    h1, h2, h3 { color: #2ea8ff !important; font-weight: 600; }

    /* Buttons */
    .stButton>button {
        background-color: #2ea8ff;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        color: white;
        font-size: 16px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #187bbf;
        transition: 0.2s;
    }

    /* Template gallery hover */
    .template-img:hover {
        transform: scale(1.04);
        transition: 0.2s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================================================
# HEADER
# ==============================================================
st.title("Object Detection via Template Matching")
st.markdown(
    """
    This application performs NCC-based template matching for object detection.
    Select a **scene**, and the system will evaluate **all templates** and automatically
    determine the correct one using **scene–template name bias** or **highest NCC score**.
    """
)


# ==============================================================
# LOAD TEMPLATES AND SCENES
# ==============================================================
db = load_templates(TEMPLATES_DIR)

scene_files = [f for f in sorted(os.listdir(SCENES_DIR))
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if len(db) == 0:
    st.error("No templates found in `data/templates/`. Cannot proceed.")
    st.stop()


# ==============================================================
# SCENE SELECTION
# ==============================================================
st.markdown("---")
st.subheader("1. Select Scene")

sel_col, _ = st.columns([1.2, 0.8])

with sel_col:
    scene_choice = st.selectbox("Choose a Scene Image",
                                ["(none)"] + scene_files,
                                index=1 if scene_files else 0)

if scene_choice != "(none)":
    scene_path = os.path.join(SCENES_DIR, scene_choice)
    scene_bgr = cv2.imread(scene_path)
    st.image(bgr_to_rgb(scene_bgr),
             caption=f"Scene: {scene_choice}",
             use_container_width=True)
else:
    scene_bgr = None


# ==============================================================
# TEMPLATE GALLERY
# ==============================================================
st.markdown("---")
st.subheader("2. Template Database (Reference Only)")

cols = st.columns(5)
for i, (name, img) in enumerate(db.items()):
    with cols[i % 5]:
        st.markdown(f"<div class='template-img'>", unsafe_allow_html=True)
        st.image(bgr_to_rgb(resize_for_display(img, 200)),
                 caption=name,
                 use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================
# RUN DETECTION
# ==============================================================
st.markdown("---")
run = st.button("Run Detection", use_container_width=True)

enable_blur = st.sidebar.checkbox("Blur Detected Region", value=True)

if run:
    if scene_bgr is None:
        st.warning("Please select a valid scene image.")
        st.stop()

    st.subheader("3. Detection Results")

    scene_key = os.path.splitext(scene_choice)[0].replace("scene_", "").lower()
    best_name, best_vis, best_boxes, best_score = None, None, [], -1.0
    matched_by_name = None

    # Evaluate all templates
    for name, templ_bgr in db.items():
        vis, boxes, score = match_template_best(scene_bgr, templ_bgr, THRESHOLD)
        if score > best_score:
            best_name, best_vis, best_boxes, best_score = name, vis, boxes, score
        if scene_key in name.lower():
            matched_by_name = (name, vis, boxes, score)

    # Bias resolution
    if matched_by_name and len(matched_by_name[2]) > 0:
        chosen_name, chosen_vis, chosen_boxes, chosen_score = matched_by_name
        method = "Matched by Scene Name"
    elif best_name and best_boxes:
        chosen_name, chosen_vis, chosen_boxes, chosen_score = best_name, best_vis, best_boxes, best_score
        method = "Highest NCC Score"
    else:
        st.warning("No detections above threshold.")
        st.stop()

    st.success(f"Detection: {chosen_name} | Method: {method} | Score: {chosen_score:.2f}")

    # Blur region
    blurred = blur_regions(chosen_vis, chosen_boxes) if enable_blur else chosen_vis

    # Save result
    save_path = save_result(blurred, stem=f"{chosen_name}_blurred")

    colA, colB = st.columns(2)
    with colA:
        st.image(bgr_to_rgb(chosen_vis), caption="Detected Region", use_container_width=True)
    with colB:
        st.image(bgr_to_rgb(blurred), caption="Blurred Output", use_container_width=True)

    st.markdown(f"Saved output: `{save_path}`")
    st.caption("Developed for CSc 8830 – Computer Vision, Georgia State University.")
