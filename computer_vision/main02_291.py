import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image
img = cv2.imread("UTC2_02.jpg")

# Get the height and width of the image
(H, W) = img.shape[:2]

# Calculate the mean pixel values
mean_pixel_values = np.average(img, axis=(0, 1))

# Create a blob from the input image
blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(W, H),
                             mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]),
                             swapRB=False, crop=False)

# Reshape the blob for display
blob_for_plot = np.moveaxis(blob[0, :, :, :], 0, 2)

# Display the original and preprocessed images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(blob_for_plot)
plt.title('Preprocessed Image (Blob)')
plt.axis('off')

plt.show()

