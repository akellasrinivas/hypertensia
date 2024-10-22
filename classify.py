import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import pandas as pd
import os

def classify_image(img, model_path='models/HR_final.h5', threshold=0.3):
    """
    Classify an image using a pre-trained model with thresholding.

    Args:
    - img: Input image (PIL image object)
    - model_path: Path to the pre-trained model (default is 'HR_sixtyfour.h5')
    - threshold: Threshold value for classification (default is 0.3)

    Returns:
    - Classification result: "Hypertensive retinopathy detected" or "Hypertensive retinopathy not detected"
    """
    img = image.load_img(img)
    # Load the pre-trained model
    model = load_model(model_path)

    # Define the input shape
    h, w = 256, 256

    # Resize the image to match the model input size
    img = img.resize((w, h))

    # Preprocess the input image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize pixel values to [0, 1]

    # Make predictions
    predictions = model.predict(img_array)

    # Check if predictions are above the threshold
    if predictions[0][0] > threshold:
        return "Hypertensive retinopathy detected"
    else:
        return "Hypertensive retinopathy not detected"

def check_image_in_csv(image_name, csv_file=r"csv\final.csv"):
    """
    Check if an image is present in the CSV file and its corresponding Hypertensive Retinopathy label.

    Args:
    - image_name: Name of the image file
    - csv_file: Path to the CSV file containing image names and labels

    Returns:
    - If image is present: "Hypertensive retinopathy detected" or "Hypertensive retinopathy not detected"
    - If image is not present: None
    """
    # Load CSV file
    image_name = os.path.basename(image_name) 
    df = pd.read_csv(csv_file)

    # Check if image_name is present in the 'Image' column
    if image_name in df['Image'].values:
        # Check the corresponding Hypertensive Retinopathy value
        hr_value = df.loc[df['Image'] == image_name, 'Hypertensive Retinopathy'].values[0]
        if hr_value == 0:
            return "Hypertensive retinopathy not detected."
        else:
            return "Hypertensive retinopathy detected."
    else:
        return None

# Example usage:
image_name = r'Image_for_classification\train\HR\00001a1e.png'
 # Example image name
csv_file = r'csv\final.csv'  # Example CSV file

# First, check in CSV file
csv_result = check_image_in_csv(image_name, csv_file)
if csv_result:
    print("Result from CSV file:", csv_result)
else:
    # If not found in CSV, classify using the model
   # Assuming image path
    model_result = classify_image(image_name)
    print("Result from model classification:", model_result)
