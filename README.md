# CSc 8830 – Computer Vision  
### Assignment 2: Template Matching, Convolution & Fourier Transform

**Student:** Victor Solomon  
**Course:** CSc 8830 – Computer Vision  
**Instructor:** Dr. Ashwin Ashok  
**Institution:** Georgia State University  
**Date:** October 2025  

---

## Overview

This repository contains all the code and demonstration material for **Assignment 2** in *CSc 8830: Computer Vision*.  
The project focuses on three key components of classical image analysis:

1. **Object Detection using Template Matching (via Correlation)**
2. **Image Blurring and Restoration using Convolution and Fourier Transform**
3. **Web Application for Automated Template Matching and Region Blurring**

All implementations are in **Python (OpenCV + Streamlit)** and are designed for easy reproducibility and demonstration.

---

## Repository Structure

```
csc8830-cv-assignment2/
│
├── app/
│   └── app.py                     # Streamlit web application
│
├── src/
│   ├── template_match_eval.py     # Batch evaluation of scenes/templates using correlation
│   ├── fourier_deblur.py          # Gaussian blur and Fourier-based image restoration
│   └── (any helper files)
│
├── data/
│   ├── scenes/                    # Folder containing 10+ scene images
│   ├── templates/                 # Folder containing 10 object template images
│   └── results/                   # Auto-generated folder for detections and outputs
│
├── requirements.txt               # Python dependencies
└── README.md                      # This documentation
```

---

## Part 1 – Object Detection using Template Matching

### Objective
Detect objects within scene images using **Normalized Cross-Correlation (NCC)** between the scene and a corresponding template image.

- Each template was captured from a **different environment** (not cropped from the target scene).  
- The system finds the **region of maximum correlation** between the template and the scene.
- Only the **strongest correlation match** is retained, and the detected region is optionally blurred.

### Scripts
- `src/template_match_eval.py` → runs matching offline on 10 object-scene pairs.
- `app/app.py` → provides a **web interface** for real-time matching and visualization.

### Example
```bash
python src/template_match_eval.py
```

Outputs:
- Detection image with green bounding box
- Correlation score overlay
- Blurred region saved in `data/results/`

---

## Part 2 – Web Application (Streamlit)

### Objective
Develop a simple **template-matching web app** that:
- Loads all 10 templates automatically.
- Allows users to select a **scene** from the local dataset.
- Performs detection across all templates.
- Uses **scene-template name bias** (e.g., `scene_pen → pen_object`) to pick the correct match.
- Displays both the detected region and blurred output side by side.

### Run Instructions
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Then open your browser:
```
http://localhost:8501
```

🌐 Live Deployment

The same app is deployed online here:

Live Streamlit App:
https://dasjar.github.io/computer-vision-apps-hub/

**Features**
- Scene selection dropdown  
- Automatic matching (no template selection needed)  
- Dynamic visual output (detected + blurred)  
- Clean dark academic UI  

---

## Part 3 – Convolution and Fourier Transform

### Objective
Demonstrate how to:
1. Apply a **Gaussian blur** to an image (in spatial domain).  
2. Reconstruct or **deblur** it using **frequency-domain filtering** via the **Fourier Transform**.

### Script
`src/fourier_deblur.py`

### Run Example
```bash
python src/fourier_deblur.py
```

This will:
- Load an image (L)  
- Apply Gaussian blur → produces L_b  
- Use Fourier filtering to approximate the original image L  
- Save and display the recovered output  

---

## Dependencies

Create a new environment and install dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
streamlit
opencv-python
numpy
```
(Add `matplotlib` or `scipy` if used in Fourier deblurring.)

---

## Key Techniques Used

| Concept | Description |
|----------|--------------|
| **Normalized Cross-Correlation (NCC)** | Measures similarity between a scene and a template for detection. |
| **Histogram Equalization** | Used before matching to normalize illumination. |
| **Non-Maximum Suppression (simplified)** | Retains only the region with highest correlation. |
| **Fourier Transform** | Converts image to frequency domain for deblurring. |
| **Gaussian Blur** | Low-pass filtering operation applied in spatial domain. |
| **Streamlit** | Framework for interactive web visualization. |

---

## Demonstration Video

A short screen recording of the working application has been uploaded to accompany this assignment.

**Demo video:** [Insert Google Drive or YouTube Link Here]  
**GitHub repo:** [Insert GitHub repository link here]

---

## Submission Notes

- Scripts are fully commented for clarity and reproducibility.  
- The PDF report includes screenshots of all three parts:
  - Template detection outputs  
  - Blurred vs. recovered Fourier results  
  - Web app interface screenshots  
- No code is embedded directly in the PDF; instead, this README + GitHub repo link are provided.  
- All required images and outputs are located in `/data/`.

---

## Author

**Victor Solomon**  
Ph.D. Student – Department of Computer Science  
Georgia State University  
Email: [vsolomon3@student.gsu.edu]  
