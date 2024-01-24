# from PIL import Image

# def to_grayscale(src_path):
#     # Check if the source is a file path, then open the image
#     if isinstance(src_path, str):
#         src = Image.open(src_path)
#     else:
#         raise TypeError('Invalid source type. Provide a file path.')

#     # Convert the image to grayscale
#     grayscale_image = src.convert('L')

#     return grayscale_image

# if __name__ == '__main__':
#     # Specify the image file path
#     filename = 'pebbles.jpg'

#     # Convert the image to black and white (grayscale)
#     grayscale_result = to_grayscale(filename)

#     # Display the original and grayscale images side by side
#     grayscale_result.show()


import cv2

def to_grayscale_opencv(src_path):
    # Read the image using OpenCV
    image = cv2.imread(src_path)

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return grayscale_image



if __name__ == '__main__':
    # Specify the image file path
    filename = 'UTC2_01.jpg'

    # Convert the image to black and white (grayscale) using OpenCV
    grayscale_result_opencv = to_grayscale_opencv(filename)
    # Save the resulting edge-detected image
    output_path = "UTC2_01_trang_den.jpg"
    cv2.imwrite(output_path, grayscale_result_opencv)

    # Display the original and grayscale images using matplotlib
    import matplotlib.pyplot as plt

    plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(1, 2, 2), plt.imshow(grayscale_result_opencv, cmap='gray'), plt.title('Grayscale Image')

    plt.show()
