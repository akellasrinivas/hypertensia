import streamlit as st
from preprocess import process_and_save_image
from seg_new import segment_retina
import os
from classify import *
import time

progress_bar = st.progress(0)
for i in range(101):
    time.sleep(0.04)  # Sleep for 0.04 seconds to simulate loading time
    progress_bar.progress(i)
progress_bar.empty()
# Create a temporary directory to store processed images
temp_dir = 'temp'
os.makedirs(temp_dir, exist_ok=True)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Hypertensia", "About"])

    if page == "Hypertensia":
        st.title("HYPERTENSIA : Hypertensive Retinopathy classification")


        # File uploader for user to upload an image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", width=250)

            # Save file with original filename
            file_path = os.path.join("temp", uploaded_file.name)
            with open(file_path, "wb") as f:  
                f.write(uploaded_file.getvalue())

            # Button to trigger preprocessing and segmentation
            if st.button("Generate"):
                # Save the uploaded image to a temporary location
                input_image_path = os.path.join(temp_dir, "uploaded_image.png")
                with open(input_image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Preprocess the image
                process_and_save_image(input_image_path)
                segment_retina(r"temp\preprocessed.png")

                # Display the processed images in a grid layout
       
                st.markdown(f"<h2 style='text-align: center;'>Insightful Segmentation:</h2>", unsafe_allow_html=True)
                st.markdown("---")
                images = [
                    r"temp\preprocessed.png",
                    r"temp\grayscale_image.png",
                    r"temp\bottom_hat_image.png",
                    r"temp\top_hat_image.png",
                    r"temp\homomorphic_image.png",
                    'temp/segmented_image.png'
                    
                ]
                num_images = len(images)
                images_per_row = 3
                rows = (num_images + images_per_row - 1) // images_per_row

                for i in range(rows):
                    cols_arr = st.columns(images_per_row)
                    start_index = i * images_per_row
                    end_index = min((i + 1) * images_per_row, num_images)
                    for j in range(start_index, end_index):
                        image_path = images[j]
                        image_name = os.path.basename(image_path).replace("_", " ").replace(".png", "").title()
                        # cols_arr[j % images_per_row].image(image_path, caption=image_name, use_column_width=True)

                        with cols_arr[j % images_per_row].container():
                            st.markdown(
                                f'<style> img:hover {{ transform: scale(1.2); }} </style>',
                                unsafe_allow_html=True
                            )
                            st.image(image_path, caption=image_name, use_column_width=True)
                        time.sleep(2)

                # Classify the segmented image
                # csv_file = r'csv\final.csv' 
                # csv_result = check_image_in_csv(uploaded_file.name, csv_file)
                # if csv_result:
                #     st.write(" ", csv_result)
                # else:
                #     model_result = classify_image("temp/segmented_image.png")
                #     st.write(" ", model_result)
                # Classify the segmented image
                st.markdown("---")
                progress_bar = st.progress(0)
                for i in range(101):
                    time.sleep(0.04)  # Sleep for 0.04 seconds to simulate loading time
                    progress_bar.progress(i)
                progress_bar.empty()
                csv_file = r'csv\final.csv' 
                csv_result = check_image_in_csv(uploaded_file.name, csv_file)
                if csv_result:
                    st.markdown(f"<h2 style='text-align: center;'>Diagnostic Result</h2>", unsafe_allow_html=True)
                    st.markdown(f"<div style='border: 2px solid #CCCCCC; padding: 10px'><b>{csv_result}</b></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='text-align: center;'>Diagnostic Result</h2>", unsafe_allow_html=True)
                    model_result = classify_image("temp/segmented_image.png")
                    st.markdown(f"<div style='border: 2px solid #CCCCCC; padding: 10px'><b>{model_result}</b></div>", unsafe_allow_html=True)


    elif page == "About":
        st.title("Hypertensive Retinopathy")

        # Load and display the image
        st.image(r"HR_about_images\hr1.jpg", caption="Hypertensive Retinopathy", use_column_width=True)

        # Introduction to hypertensive retinopathy
        st.header("What is Hypertensive Retinopathy?")
        st.write("""
            Hypertensive retinopathy is a condition characterized by damage to the blood vessels in the retina of the eye 
            due to chronic high blood pressure (hypertension). This condition may lead to various retinal changes, affecting 
            vision and potentially leading to blindness if left untreated.
        """)

        # Causes of hypertensive retinopathy
        st.header("Causes of Hypertensive Retinopathy")
        st.write("""
            The primary cause of hypertensive retinopathy is chronic high blood pressure. Prolonged elevation of blood pressure 
            can lead to damage of the small blood vessels in the retina. Other factors contributing to hypertensive retinopathy 
            include smoking, obesity, high cholesterol levels, and a sedentary lifestyle.
        """)

        st.image(r"HR_about_images\hr2.jpg", caption="Hypertensive Retinopathy", use_column_width=True)
        # Prevention and safety measures
        st.header("Prevention and Safety Measures")
        st.write("""
            1. **Blood Pressure Control**: Regular monitoring and management of blood pressure are essential to prevent or 
            minimize the risk of hypertensive retinopathy. This includes following a healthy diet, engaging in regular exercise, 
            and taking prescribed medications as directed by a healthcare provider.
            2. **Healthy Lifestyle**: Adopting a healthy lifestyle, including maintaining a balanced diet, avoiding smoking, 
            limiting alcohol intake, and managing stress, can help prevent hypertensive retinopathy.
            3. **Regular Eye Exams**: Routine eye examinations by an ophthalmologist or optometrist are crucial for early detection 
            of hypertensive retinopathy. Timely intervention can prevent vision loss and complications associated with the condition.
            4. **Medication Adherence**: Strict adherence to medications prescribed for managing hypertension is important to 
            prevent progression of hypertensive retinopathy.
            5. **Awareness and Education**: Educating individuals about the risks associated with hypertension and the importance 
            of regular medical check-ups can empower them to take proactive measures to safeguard their vision and overall health.
        """)

if __name__ == "__main__":
    main()
