import cv2
import numpy as np
from matplotlib import pyplot as plt

# Custom CropLayer
class CropLayer(object):
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of
        # the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]

# The pre-trained model that OpenCV uses has been trained in Caffe framework
# Download from the link above
# Model pre-training mà OpenCV sử dụng đã được train trong Caffe framework
# Tải xuống từ liên kết trên

protoPath = "E:\\0_NCKH_XETUHANH\\ComputerVision\\ComputerVision\\291\\deploy.prototxt"
modelPath = "E:\\0_NCKH_XETUHANH\\ComputerVision\\ComputerVision\\291\\hed_pretrained_bsds.caffemodel"

# Read the model using cv2.dnn.readNet
# Đọc mô hình bằng cv2.dnn.readNet
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Register custom crop layer with the model
# Đăng ký lớp cắt tùy chỉnh với mô hình
cv2.dnn_registerLayer("Crop", CropLayer)

# Open a connection to the webcam (change the parameter to the appropriate camera index)
# Mở kết nối tới webcam (thay đổi thông số về chỉ số camera phù hợp)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
# Kiểm tra xem webcam đã mở thành công chưa
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Khởi tạo biến cờ để kiểm tra việc thoát khỏi vòng lặp 
exit_flag = False

while not exit_flag:
    # Read a frame from the webcam
    # Đọc khung hình từ webcam
    ret, frame = cap.read()

    # Check if the frame was read successfully
    # Kiểm tra xem frame đã được đọc thành công chưa
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Resize the frame to match the input size expected by the network
    # Thay đổi kích thước khung để phù hợp với kích thước đầu vào mà mạng mong đợi
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(256, 256), mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False)

    # Set the input to the network
    # Đặt đầu vào cho mạng
    net.setInput(blob)

    # Perform a forward pass to get the output
    # Thực hiện chuyển tiếp để nhận đầu ra
    edges = net.forward()

    # Process the output to get the binary edges
    # Xử lý đầu ra để lấy các cạnh nhị phân
    edges = edges[0, 0]
    edges = cv2.resize(edges, (frame.shape[1], frame.shape[0]))
    edges = cv2.threshold(edges, 0.1, 1.0, cv2.THRESH_BINARY)[1]

    # Apply the edges on the original frame
    # Áp dụng các cạnh trên khung ban đầu
    result = (frame * edges[:, :, np.newaxis]).astype(np.uint8)

    # Display the original frame and the result using matplotlib
    # Hiển thị frame gốc và kết quả bằng matplotlib
    plt.subplot(121), plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), plt.title('Original Frame')
    plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Object Extraction')

    # Adjust layout and show the plot
    # Điều chỉnh bố cục và hiển thị cốt truyện
    plt.tight_layout()
    plt.pause(0.1)  # Add a small delay to allow the plot to update # Thêm một độ trễ nhỏ để cho phép cập nhật cốt truyện
    plt.clf()  # Clear the plot for the next iteration # Xóa cốt truyện cho lần lặp tiếp theo

            # Chờ 1ms và kiểm tra xem có phím nào được nhấn không
    key = cv2.waitKey(1)

    # Kiểm tra xem phím ESC được nhấn không
    if key == 27:  # 27 là mã ASCII của phím ESC
        exit_flag = True


# Release the webcam and close all windows
# Nhả webcam và đóng tất cả các cửa sổ
cap.release()
cv2.destroyAllWindows()
