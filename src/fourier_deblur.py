"""
CSc 8830 - Assignment 2
Part B: Convolution and Fourier Transform
-----------------------------------------
This script demonstrates image blurring and restoration using
Fourier-domain filtering (inverse and Wiener deconvolution).

It supports both grayscale and RGB images automatically.
Outputs:
 - blurred.jpg
 - recovered_inverse.jpg
 - recovered_wiener.jpg
"""

import cv2
import numpy as np
import os

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
img_path = "/data/users2/vsolomon3/csc8830-cv-assignment2/data/scenes/scene_earbud.jpg"
out_dir = "data/results"
kernel_size = 61    # Gaussian blur kernel size
sigma = 15       # Standard deviation of Gaussian
K = 0.01            # Wiener filter constant

os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------
def wiener_filter(F_blur, F_kernel, K=0.01):
    """Perform Wiener deconvolution in the frequency domain."""
    H_conj = np.conj(F_kernel)
    denominator = np.abs(F_kernel) ** 2 + K
    return (H_conj / denominator) * F_blur


def fourier_deblur_channel(channel, kernel_size, sigma, K):
    """Blur and deblur a single image channel using Fourier transform."""
    # Apply Gaussian blur in spatial domain
    blurred = cv2.GaussianBlur(channel, (kernel_size, kernel_size), sigma)

    # Create padded kernel the same size as image
    H = cv2.getGaussianKernel(kernel_size, sigma)
    H = H @ H.T
    H_padded = np.zeros_like(channel, dtype=np.float32)
    center_r, center_c = channel.shape[0] // 2, channel.shape[1] // 2
    kh, kw = H.shape
    H_padded[center_r - kh//2:center_r + kh//2 + 1,
             center_c - kw//2:center_c + kw//2 + 1] = H
    H_padded = np.fft.ifftshift(H_padded)

    # FFT of blurred image and kernel
    F_blur = np.fft.fft2(blurred)
    F_kernel = np.fft.fft2(H_padded)

    # Inverse filter
    F_inv = np.divide(F_blur, F_kernel, out=np.zeros_like(F_blur), where=np.abs(F_kernel) > 1e-3)
    recovered_inv = np.abs(np.fft.ifft2(F_inv))

    # Wiener filter
    F_wiener = wiener_filter(F_blur, F_kernel, K)
    recovered_wiener = np.abs(np.fft.ifft2(F_wiener))

    # Normalize
    blurred = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    recovered_inv = cv2.normalize(recovered_inv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    recovered_wiener = cv2.normalize(recovered_wiener, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return blurred, recovered_inv, recovered_wiener


# ---------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------
print("[INFO] Loading input image...")
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

print(f"[INFO] Image loaded with shape {img.shape}")

# Handle both grayscale and color
if len(img.shape) == 2:  # grayscale
    blurred, rec_inv, rec_wiener = fourier_deblur_channel(img, kernel_size, sigma, K)
else:  # color
    print("[INFO] Processing color image (per-channel)...")
    channels = cv2.split(img)
    blurred_channels, inv_channels, wiener_channels = [], [], []

    for ch in channels:
        b, ri, rw = fourier_deblur_channel(ch, kernel_size, sigma, K)
        blurred_channels.append(b)
        inv_channels.append(ri)
        wiener_channels.append(rw)

    blurred = cv2.merge(blurred_channels)
    rec_inv = cv2.merge(inv_channels)
    rec_wiener = cv2.merge(wiener_channels)

# Save results
cv2.imwrite(os.path.join(out_dir, "blurred.jpg"), blurred)
cv2.imwrite(os.path.join(out_dir, "recovered_inverse.jpg"), rec_inv)
cv2.imwrite(os.path.join(out_dir, "recovered_wiener.jpg"), rec_wiener)

print("[INFO] Saved:")
print(f" - Blurred image → {os.path.join(out_dir, 'blurred.jpg')}")
print(f" - Recovered (Inverse) → {os.path.join(out_dir, 'recovered_inverse.jpg')}")
print(f" - Recovered (Wiener) → {os.path.join(out_dir, 'recovered_wiener.jpg')}")
print("[DONE]")
