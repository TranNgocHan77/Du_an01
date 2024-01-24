import cv2
from matplotlib import pyplot as plt
import numpy as np

class CropLayer(object):
    def __init__(self, params, blobs):
        # Initialize the starting and ending (x, y)-coordinates for cropping
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # The crop layer will receive two inputs:
        # Crop the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # Compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        # Return the shape of the volume (actual cropping happens during the forward pass)
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # Use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY, self.startX:self.endX]]

# Load the pre-trained model from Caffe framework
protoPath = "deploy.prototxt"
modelPath = "hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Register the custom crop layer with the model
cv2.dnn_registerLayer("Crop", CropLayer)

# Load the input image and get its dimensions for defining the blob
img = cv2.imread("pebbles.jpg")
plt.imshow(img)
(H, W) = img.shape[:2]

# Create a blob from the input image (preprocessing)
mean_pixel_values = np.average(img, axis=(0, 1))
blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(W, H),
                             mean=(105, 117, 123),
                             swapRB=False, crop=False)

# View the image after preprocessing (blob)
blob_for_plot = np.moveaxis(blob[0, :, :, :], 0, 2)
plt.imshow(blob_for_plot)

# Set the blob as the input to the network and perform a forward pass
# to compute the edges
net.setInput(blob)
hed = net.forward()
hed = hed[0, 0, :, :]  # Drop the other axes
hed = (255 * hed).astype("uint8")  # Rescale to 0-255

# # Save the resulting edge-detected image
# output_path = "pebbles_gray.jpg"
# cv2.imwrite(output_path, hed)

plt.imshow(hed, cmap='gray')
plt.show()
