import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Görselleri yükle
img1 = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("image2.jpg", cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise ValueError("Görseller yüklenemedi.")

# ---------------- ORB ----------------
orb = cv2.ORB_create(nfeatures=500)

start_orb = time.time()
kp1_orb, des1_orb = orb.detectAndCompute(img1, None)
kp2_orb, des2_orb = orb.detectAndCompute(img2, None)
time_orb = time.time() - start_orb

bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf_orb.match(des1_orb, des2_orb)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)
mean_dist_orb = np.mean([m.distance for m in matches_orb])

# ---------------- SIFT ----------------
sift = cv2.SIFT_create(nfeatures=500)

start_sift = time.time()
kp1_sift, des1_sift = sift.detectAndCompute(img1, None)
kp2_sift, des2_sift = sift.detectAndCompute(img2, None)
time_sift = time.time() - start_sift

# SIFT için de crossCheck=True matcher kullanalım (daha adil)
bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_sift = bf_sift.match(des1_sift, des2_sift)
matches_sift = sorted(matches_sift, key=lambda x: x.distance)
mean_dist_sift = np.mean([m.distance for m in matches_sift])

# ---------------- Görselleri çiz ----------------
img_match_orb = cv2.drawMatches(img1, kp1_orb, img2, kp2_orb, matches_orb[:30], None, flags=2)
img_match_sift = cv2.drawMatches(img1, kp1_sift, img2, kp2_sift, matches_sift[:30], None, flags=2)

plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.imshow(img_match_orb)
plt.title(f"ORB - Eşleşme: {len(matches_orb)}, Süre: {time_orb:.3f}s, Ortalama Mesafe: {mean_dist_orb:.2f}")

plt.subplot(2, 1, 2)
plt.imshow(img_match_sift)
plt.title(f"SIFT - Eşleşme: {len(matches_sift)}, Süre: {time_sift:.3f}s, Ortalama Mesafe: {mean_dist_sift:.2f}")

plt.tight_layout()
plt.show()
