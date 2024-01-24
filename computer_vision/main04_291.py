import cv2
import numpy as np
from matplotlib import pyplot as plt

# Phần mã này tải mô hình HED được đào tạo trước bằng cách sử dụng các tệp triển khai.prototxt (kiến trúc mô hình) và hed_pretrain_bsds.caffemodel (trọng lượng mô hình).
# Load the HED model
prototxt_path = "deploy.prototxt"  # Replace with the correct path to the deploy.prototxt file
caffemodel_path = "hed_pretrained_bsds.caffemodel"  # Replace with the correct path to the pretrained model

net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Tải và xử lý trước hình ảnh đầu vào:
# Mã này đọc hình ảnh đầu vào ("pebbles.jpg") và tính chiều cao ( H) và chiều rộng ( W) của nó.
# Nó tính toán các giá trị pixel trung bình của hình ảnh và sau đó tạo một đốm màu từ hình ảnh bằng cv2.dnn.blobFromImagechức năng của OpenCV. Blob này sẽ là đầu vào của mạng lưới thần kinh.
# Load and preprocess the input image
img = cv2.imread("UTC2_02.jpg")
(H, W) = img.shape[:2]

mean_pixel_values = np.average(img, axis=(0, 1))
blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(W, H),
                             mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]),
                             swapRB=False, crop=False)

# Chuyển tiếp qua mạng
# Set the blob as the input to the network and perform a forward pass to compute the edges
# Blob được đặt làm đầu vào của mạng ( net.setInput(blob)) và quá trình chuyển tiếp được thực hiện để tính toán các cạnh bằng mô hình HED.
# Đầu ra ( hed) được trích xuất từ ​​kết quả chuyển tiếp. Các trục được điều chỉnh và các giá trị được thay đổi tỷ lệ thành phạm vi [0, 255].
net.setInput(blob)
hed = net.forward()
hed = hed[0, 0, :, :]  # Drop the other axes
hed = (255 * hed).astype("uint8")  # Rescale to 0-255

#Độ mờ và ngưỡng Gaussian:
# Load segmented binary image, Gaussian blur, grayscale, Otsu's threshold
# Các cạnh được tính toán ( hed) được làm mờ bằng bộ lọc Gaussian để làm mịn hình ảnh.
# Ngưỡng của Otsu được áp dụng để thu được ảnh nhị phân ( thresh) trong đó các cạnh được nhấn mạnh.
blur = cv2.GaussianBlur(hed, (3, 3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# trình bày kết quả hiển hị ra màn hình.
# Display the original image, computed edges, and segmented binary image
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(hed, cmap='gray')
plt.title('Computed Edges')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(thresh, cmap='gray')
plt.title('Segmented Binary Image')
plt.axis('off')

plt.show()
