import cv2
import os 
from ietk import methods

def process_and_save_image(input_image_path, output_dir= 'temp/'):
    def normalize(img):
        # Convert pixel values to the range [0, 1]
        img = img.astype('float32')
        img /= 255
        return img

    def enhance(img):
        # Enhance the image using methods from the ICIAR 2020 paper
        enhanced_img = methods.brighten_darken(img, 'A+B+X')
        enhanced_img = methods.sharpen(enhanced_img)
        return enhanced_img

    def denormalize(img):
        # Convert pixel values back to the range [0, 255]
        denorm_img = img * 255.0
        return denorm_img

    # Read the image from file
    input_image = cv2.imread(input_image_path)
    if input_image is None:
        print(f"Error: Unable to read image from '{input_image_path}'")
        return None
    
    # Process the image
    processed_image = input_image.copy()
    processed_image = normalize(processed_image)
    processed_image = enhance(processed_image)
    processed_image = denormalize(processed_image)
    
    
    
    # Generate the output filename
    output_filename = "preprocessed.png"
    
    # Construct the output file path
    output_path = os.path.join(output_dir, output_filename)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write the processed image to file
    cv2.imwrite(output_path, processed_image)
    print(f"Processed image saved to '{output_path}'")
    
    return processed_image

# Example usage:
if __name__ == "__main__":
    # Single image path
    image_path = r'Image_for_classification\train\HR\0000d073.png'

    # Process and save the image
    preprocessed_image = process_and_save_image(image_path)
