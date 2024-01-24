import cv2
from matplotlib import pyplot as plt
import numpy as np

# Custom CropLayer implementation
class CropLayer(object):
    def __init__(self, params, blobs):
        # Initialize starting and ending (x, y)-coordinates for cropping
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # Crop the first input blob to match the shape of the second one
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # Compute starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        # Return the shape of the volume (actual cropping happens during forward pass)
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # Use derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY, self.startX:self.endX]]

# Load pre-trained HED model from Caffe framework
protoPath = "deploy.prototxt"
modelPath = "hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Register custom crop layer with the model
cv2.dnn_registerLayer("Crop", CropLayer)

# Load input image and get its dimensions for defining the blob
img = cv2.imread("pebbles.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Display the image in RGB

# Create a blob from the input image (preprocessing)
mean_pixel_values = np.average(img, axis=(0, 1))
blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(img.shape[1], img.shape[0]),
                             mean=(105, 117, 123), swapRB=False, crop=False)

# Set the blob as the input to the network and perform a forward pass to compute the edges
net.setInput(blob)
hed = net.forward()
hed = hed[0, 0, :, :]  # Drop the other axes
hed = (255 * hed).astype("uint8")  # Rescale to 0-255

# Display the HED result
plt.imshow(hed, cmap='gray')  # Display the result in grayscale
plt.show()
