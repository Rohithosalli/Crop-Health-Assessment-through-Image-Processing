# streamlit run c:\Users\rohan\OneDrive\Documents\code_VS\minor_web\main.py

import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("C:\\Users\\rohan\\OneDrive\\Documents\\code_VS\\minor_web\\Improved_trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)  # Return index of max element
    return result_index 

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("Crop Health Assessment through Image Processing")
    image_path = "C:\\Users\\rohan\\OneDrive\\Documents\\code_VS\\minor_web\\home_pgi.JPG"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    
    # PROBLEM STATEMENT
    #### In the traditional methods of crop health assessment often rely on visual inspection by farmers, which can be subjective and time-consuming. Early detection of diseases or nutrient deficiencies is crucial for taking timely corrective actions and minimizing crop losses. To address this issue there is a critical need for the development of a crop health assessment.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown(""" 
    # About Dataset
    ### This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
    
    # Content
    ### 1. Train 70295 images
    ### 2. Validation 17572 image
    ### 3. Test 33 images
    
    # Model Building:
    ####  1. Loads and Processes images in RGB format in batches of 32. Uses categorical labels and  Resizes images to 128x128 pixels.
    ####  2. Conv2D Layers
    ####  3. BatchNormalization
    ####  4. MaxPooling2D
    ####  5. Dropout
    ####  6. Flatten
    ####  7. Dense Layers
    ####  8. ReduceLROnPlateau
    ####  9. Model trains for 15 epochs.
    """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            
            # Define class names
            class_name = [
                'Apple___Apple_scab', 'Apple__Black_rot', 'Apple___Cedar_apple_rust',
                'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
                'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
                'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            st.success(f"Model is predicting it as {class_name[result_index]}")
