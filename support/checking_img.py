#this file is for checking if the images are being read correctly from the dataset. 
# It uses OpenCV to read the image and Matplotlib to display it. Make sure to adjust 
# the path to the image if necessary.
#in terminal: cd support -> python checking_img.py
import cv2
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(BASE_DIR, "data", "s1", "1.pgm")

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap="gray")
plt.title("Face Image")
plt.axis("off")
plt.show()