import cv2
import numpy as np

def segment_retina(image_path, output_path='temp/segmented_image.png', kernel_size=(30, 30), gamma=1, cutoff=0.5, lower_threshold=100, upper_threshold=200):
    """
    Segment retinal images using a combination of preprocessing steps.

    Args:
    - image_path: Path to the retinal color image
    - output_path: Path to save the segmented image
    - kernel_size: Size of the structuring element for morphological operations
    - gamma: Parameter for homomorphic filtering
    - cutoff: Cutoff frequency for homomorphic filtering
    - lower_threshold: Lower threshold value for binarization
    - upper_threshold: Upper threshold value for binarization

    Returns:
    - Segmented binary image
    """
    # Read the color image
    color_img = cv2.imread(image_path)
    if color_img is None:
        print(f"Error: Unable to read image from '{image_path}'")
        return None
    
    # Convert color image to grayscale
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # Define structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Perform bottom hat transform to remove uneven illumination
    bottom_hat_img = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)

    # Perform top hat transform to enhance contrast
    top_hat_img = cv2.morphologyEx(gray_img, cv2.MORPH_TOPHAT, kernel)

    # Calculate the difference between top hat and bottom hat images
    uneven_removed_img = top_hat_img - bottom_hat_img

    # Perform homomorphic filtering to remove uneven illumination
    rows, cols = uneven_removed_img.shape
    img_float = np.float32(uneven_removed_img)
    img_log = np.log1p(img_float)
    img_fft = np.fft.fft2(img_log)
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    center_u, center_v = cols // 2, rows // 2
    d_uv = np.sqrt((u - center_u) ** 2 + (v - center_v) ** 2)
    h_uv = (gamma - 1) * (1 - np.exp(-cutoff * (d_uv ** 2))) + 1
    filtered_fft = h_uv * img_fft
    img_filtered_log = np.real(np.fft.ifft2(filtered_fft))
    img_filtered = np.exp(img_filtered_log) - 1
    homomorphic_img = np.uint8(img_filtered)

    # Perform double threshold segmentation
    _, segmented_img = cv2.threshold(homomorphic_img, lower_threshold, 255, cv2.THRESH_BINARY)
    _, segmented_img = cv2.threshold(segmented_img, upper_threshold, 255, cv2.THRESH_BINARY_INV)

    # Save the segmented image
    cv2.imwrite(output_path, segmented_img)

    return segmented_img

# Example usage:
if __name__ == "__main__":
    # Provide the path to your retinal color image
    image_path = r"temp\preprocessed_0000d073.png"

    # Segment the retina and save the segmented image
    segmented_image = segment_retina(image_path)
