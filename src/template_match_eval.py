"""
CSc 8830 - Assignment 2
Object Detection using Template Matching through Correlation
-------------------------------------------------------------
Performs 1-to-1 matching between scenes and their corresponding templates.

Enhancements:
 - Keeps only the region with the highest correlation score.
 - Displays the correlation score near the detected bounding box.
 - Draws thicker green boundaries for clearer visualization.
 - Saves only detection images (no heatmaps or blurred versions).
"""

import cv2
import numpy as np
import os

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
scene_dir = "data/scenes"
template_dir = "data/templates"
out_dir = "data/results"
threshold = 0.3  # correlation threshold (adjust 0.25–0.5)

os.makedirs(out_dir, exist_ok=True)


# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------
def match_template_best(scene, templ, threshold=0.3):
    """Find only the best match (highest correlation) using NCC."""
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)

    # Normalize lighting to improve robustness
    scene_gray = cv2.equalizeHist(scene_gray)
    templ_gray = cv2.equalizeHist(templ_gray)

    # Normalized Cross-Correlation
    res = cv2.matchTemplate(scene_gray, templ_gray, cv2.TM_CCOEFF_NORMED)

    # Find global maximum correlation value and its location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If correlation is too low, skip
    if max_val < threshold:
        return scene.copy(), [], max_val

    # Draw rectangle around best match
    h, w = templ_gray.shape[:2]
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    vis = scene.copy()
    cv2.rectangle(vis, top_left, bottom_right, (0, 255, 0), 15)  # thicker line

    # Add correlation score label
    label = f"{max_val:.2f}"
    text_pos = (top_left[0], top_left[1] - 10 if top_left[1] - 10 > 20 else top_left[1] + h + 20)
    cv2.putText(vis, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3, cv2.LINE_AA)

    return vis, [(*top_left, w, h)], max_val


# ---------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------
print("[INFO] Searching for scene-template pairs...")
scenes = [f for f in os.listdir(scene_dir) if f.lower().startswith("scene_")]
log_lines = []

for scene_file in scenes:
    scene_base = scene_file.replace("scene_", "").split(".")[0]
    templ_file = f"{scene_base}_object.jpg"
    scene_path = os.path.join(scene_dir, scene_file)
    templ_path = os.path.join(template_dir, templ_file)

    if not os.path.exists(templ_path):
        print(f"[WARN] No matching template for {scene_file} → skipping.")
        log_lines.append(f"{scene_file}: no matching template found")
        continue

    scene = cv2.imread(scene_path)
    templ = cv2.imread(templ_path)
    if scene is None or templ is None:
        print(f"[WARN] Could not read {scene_file} or {templ_file}. Skipping.")
        continue

    print(f"\n[INFO] Matching {scene_file} ↔ {templ_file}")
    vis, boxes, score = match_template_best(scene, templ, threshold)

    if len(boxes) == 0:
        print(f" -> No detections (max corr = {score:.2f}), skipping save.")
        log_lines.append(f"{scene_file}: 0 detections (max corr={score:.2f})")
        continue

    print(f" -> Detection found (corr = {score:.2f})")
    log_lines.append(f"{scene_file}: 1 detection (corr={score:.2f})")

    out_path = os.path.join(out_dir, f"{scene_base}_detections.jpg")
    cv2.imwrite(out_path, vis)

# Write log file
log_path = os.path.join(out_dir, "evaluation_log.txt")
with open(log_path, "w") as f:
    f.write("\n".join(log_lines))

print("\n[INFO] Evaluation complete.")
print(f"[INFO] Log saved at {log_path}")
print("[DONE]")
