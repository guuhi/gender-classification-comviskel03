import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_crop_face(img, margin=0.1):
    """
    Detects faces in an image and crops the face with minimal margin to zoom in on the face.

    Parameters:
        img (PIL.Image): The input image.
        margin (float): Smaller margin for tighter cropping (default 10%).

    Returns:
        PIL.Image: Cropped face with minimal margin or original image if no face is detected.
    """
    # Convert PIL image to OpenCV format (numpy array)
    img_cv = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Get the largest detected face
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

        # Reduce margin for tighter cropping
        margin_x = int(w * margin)
        margin_y = int(h * margin)

        x_new = max(0, x + margin_x)
        y_new = max(0, y + margin_y)
        x_end = min(img_cv.shape[1], x + w - margin_x)
        y_end = min(img_cv.shape[0], y + h - margin_y)

        # Crop the expanded face region
        face_img = img.crop((x_new, y_new, x_end, y_end))
        return face_img
    else:
        return None  # Return None if no face is detected

def predict_gender(img):
    cropped_img = detect_and_crop_face(img)  # Detect and crop the face
    isCropped = True # Flag
    
    if cropped_img is None:
        cropped_img = img  # If no face is detected, return the original image instead
        isCropped = False # Flag

    cropped_img = cropped_img.resize((224, 224))  # Resize the face image to the model input size
    img_array = image.img_to_array(cropped_img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image to [0, 1]

    # Run inference with TFLite model
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Gender Mapping: 0 for Male, 1 for Female
    gender = 1 if prediction > 0.5 else 0  # Female = 1, Male = 0
    confidence = prediction if gender == 1 else 1 - prediction  # Confidence based on prediction

    return gender, confidence, cropped_img, isCropped  # Return cropped image along with prediction


# Streamlit interface
st.title("Kel03_Gender Classification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_pil = Image.open(uploaded_file)  # Open uploaded image
    gender, confidence, cropped_img, isCropped = predict_gender(image_pil)  # Predict gender

    # Check if no face is detected (cropped_img is the original image in this case)
    if isCropped == False:
        st.warning("⚠️ No face detected. This might affect the predictions.")
    
    # Gender display mapping
    gender_label = "Female" if gender == 1 else "Male"

    # Display results
    st.subheader(f"Prediction: {gender_label} ({round(confidence * 100, 2)}%)")
    
    # Use columns to display both images in one row
    col1, col2 = st.columns(2)
    with col1:
        st.image(cropped_img, caption="Cropped Face", width=300)  # Resize the cropped face image to fit
    with col2:
        st.image(image_pil, caption="Uploaded Image", use_container_width=True)  # Display original image with container width

