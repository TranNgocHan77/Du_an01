import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the HED model
prototxt_path = "deploy.prototxt"  # Provide the correct path to the deploy.prototxt file
caffemodel_path = "hed_pretrained_bsds.caffemodel"  # Provide the correct path to the pretrained model

net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Function for image preprocessing
def preprocess_image(image_path):
    # Load an image
    img = cv2.imread(image_path)

    # Get the height and width of the image
    (H, W) = img.shape[:2]

    # Calculate the mean pixel values
    mean_pixel_values = np.average(img, axis=(0, 1))

    # Create a blob from the input image
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(W, H),
                                 mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]),
                                 swapRB=False, crop=False)

    return img, blob, (H, W)

# Load the image and preprocess it
image_path = "UTC2_02.jpg"
img, blob, (H, W) = preprocess_image(image_path)

# Set the blob as the input to the network and perform a forward pass to compute the edges
net.setInput(blob)
hed = net.forward()
hed = hed[0, 0, :, :]  # Drop the other axes
hed = (255 * hed).astype("uint8")  # Rescale to 0-255

# Display the original image and the computed edges
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(hed, cmap='gray')
plt.title('Computed Edges')
plt.axis('off')

plt.show()
