import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('models/HR_sixtyfour.h5')

# Define the input shape
h, w = 256, 256
input_shape = (h, w, 3)

# Folder containing images to classify
folder_path = r"Image_for_classification\train\HR_mask"

# Set a threshold for classification
threshold = 0.5  # You can adjust this threshold based on your preference and evaluation

# Loop through all images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):  # Assuming images are PNG format, you can adjust this if needed
        # Load and preprocess the input image
        img_path = os.path.join(folder_path, filename)
        img = image.load_img(img_path, target_size=(h, w))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.  # Normalize pixel values to [0, 1]

        # Make predictions
        predictions = model.predict(img_array)

        # Classify based on the threshold
        if predictions[0][0] >= threshold:
            print(f"{filename}: Hypertensive retinopathy detected.")
        else:
            print(f"{filename}: No hypertensive retinopathy detected.")
