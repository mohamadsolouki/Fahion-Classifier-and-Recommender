import os
import cv2
import json
import pickle
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model # type: ignore

# --- CONFIGURATIONS ---
data_path = "data/images/"
feat_extractor = load_model('recommendation/feature_extractor_model.keras')
pca = pickle.load(open('recommendation/pca_model.pkl', 'rb'))
all_features_pca = np.load('recommendation/all_features_pca.npy')
top_similarities = json.load(open('recommendation/top_similarities.json'))
imgs_model_width, imgs_model_height = 299, 299

# --- LOAD CLASSIFICATION MODEL ---
class_model = load_model('models/best_model_Xception.keras')
class_labels = ['Backpacks' 'Belts' 'Bra' 'Briefs' 'Capris' 'Caps' 'Casual Shoes'
 'Clutches' 'Cufflinks' 'Deodorant' 'Dresses' 'Dupatta' 'Earrings' 'Flats'
 'Flip Flops' 'Formal Shoes' 'Handbags' 'Heels' 'Innerwear Vests'
 'Jackets' 'Jeans' 'Kajal and Eyeliner' 'Kurtas' 'Kurtis' 'Leggings'
 'Lip Gloss' 'Lipstick' 'Nail Polish' 'Necklace and Chains' 'Night suits'
 'Nightdress' 'Pendant' 'Perfume and Body Mist' 'Ring' 'Sandals' 'Sarees'
 'Scarves' 'Shirts' 'Shorts' 'Skirts' 'Socks' 'Sports Shoes' 'Sunglasses'
 'Sweaters' 'Sweatshirts' 'Ties' 'Tops' 'Track Pants' 'Trousers' 'Trunk'
 'Tshirts' 'Tunics' 'Wallets' 'Watches']

# --- HELPER FUNCTIONS ---

def resize_with_padding(image, target_height, target_width):
    """Resizes the image to the target size with padding."""
    height, width = tf.shape(image)[0], tf.shape(image)[1]

    if tf.equal(height, 0) or tf.equal(width, 0):
        return tf.zeros([target_height, target_width, 3], dtype=tf.float32)

    scale = tf.minimum(
        tf.cast(target_width, tf.float32) / tf.cast(width, tf.float32),
        tf.cast(target_height, tf.float32) / tf.cast(height, tf.float32),
    )
    new_height = tf.cast(tf.cast(height, tf.float32) * scale, tf.int32)
    new_width = tf.cast(tf.cast(width, tf.float32) * scale, tf.int32)
    resized_image = tf.image.resize(image, [new_height, new_width])
    padded_image = tf.image.pad_to_bounding_box(
        resized_image,
        (target_height - new_height) // 2,
        (target_width - new_width) // 2,
        target_height,
        target_width,
    )
    return padded_image

def preprocess_image(img, target_size):
    img = tf.image.decode_image(img, channels=3)
    print(f"Original image shape: {img.shape}")
    img = resize_with_padding(img, target_size[0], target_size[1])
    print(f"Resized image shape: {img.shape}")
    img = tf.keras.applications.xception.preprocess_input(img)
    print(f"Preprocessed image shape: {img.shape}")
    print(f"Preprocessed image min/max values: {tf.reduce_min(img)}/{tf.reduce_max(img)}")
    return img

def predict_image_class(img_bytes):
    """Predicts the class of a given image using the classification model."""
    img = preprocess_image(img_bytes, (299, 299))
    img = tf.expand_dims(img, 0)
    predictions = class_model.predict(img)
    predicted_class = class_labels[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

def load_and_preprocess_image(img_path):
    """Loads and preprocesses an image for the recommendation model."""
    img = tf.io.read_file(img_path)
    img = preprocess_image(img, (imgs_model_width, imgs_model_height))
    return tf.expand_dims(img, 0)

def retrieve_most_similar_products(given_img, top_n=5):
    if given_img in top_similarities:
        closest_imgs = top_similarities[given_img][:top_n]
        print(f"Using pre-calculated similarities for {given_img}")
    else:
        print(f"Calculating similarities in real-time for {given_img}")
        img_features = feat_extractor.predict(load_and_preprocess_image(given_img))
        print(f"Extracted features shape: {img_features.shape}")
        img_features_pca = pca.transform(img_features.reshape(1, -1))
        print(f"PCA transformed features shape: {img_features_pca.shape}")
        similarities = cosine_similarity(img_features_pca, all_features_pca).flatten()
        print(f"Calculated similarities shape: {similarities.shape}")
        top_indices = np.argsort(-similarities)[:top_n]
        image_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".jpg")]
        closest_imgs = [(image_files[i], float(similarities[i])) for i in top_indices]

    return closest_imgs

def image_classification_page():
    st.title("Image Classification and Recommendation")
    
    st.write("""
    Upload an image of clothing for classification and product recommendations.
    
    Instructions:
    1. Use the file uploader below to select an image.
    2. Click the 'Predict and Recommend' button to process the image.
    3. View the predicted class, confidence, and recommended products.
    """)

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_column_width=False, width=299, clamp=True)

        if st.button('Predict and Recommend'):
            with st.spinner("Processing..."):
                img_bytes = uploaded_file.getvalue()
                predicted_class, confidence = predict_image_class(img_bytes)
                
                with col2:
                    st.subheader("Classification Results")
                    st.success(f'Predicted Class: {predicted_class}')
                    st.info(f'Confidence: {confidence}%')

                original_filename = uploaded_file.name
                temp_img_path = os.path.join("data/images/", original_filename)
                with open(temp_img_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                retrieved = retrieve_most_similar_products(temp_img_path, top_n=5)

                st.subheader("Recommended Products")
                st.write("Here are 5 similar products based on your uploaded image:")

                col1, col2, col3, col4, col5 = st.columns(5)
                columns = [col1, col2, col3, col4, col5]

                for i, (img_path, score) in enumerate(retrieved):
                    with columns[i]:
                        img = Image.open(img_path)
                        st.image(img, caption=f"Score: {score:.2f}", use_column_width=True)

# --- REAL-TIME CLASSIFICATION PAGE ---
def real_time_classification_page():
    st.title("Real-time Clothing Classification")
    
    st.write("""
    Use your webcam for real-time clothing classification.
    
    Instructions:
    1. Make sure your webcam is connected and functioning.
    2. Click the 'Start Real-time Classification' button below.
    3. Point your webcam at different clothing items.
    4. The predicted class and confidence will be displayed on the video feed.
    5. Press 'q' on your keyboard to stop the classification and close the window.
    """)

    start_button = st.button("Start Real-time Classification")

    if start_button:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera. Please check your camera connection.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_preprocessed = preprocess_image(tf.image.encode_jpeg(frame_rgb), (299, 299))
            frame_array = tf.expand_dims(frame_preprocessed, 0)

            predictions = class_model.predict(frame_array)
            predicted_class = class_labels[np.argmax(predictions[0])]
            confidence = round(100 * np.max(predictions[0]), 2)

            cv2.putText(frame, f"{predicted_class} ({confidence}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Real-time Classification", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# --- STREAMLIT APP ---
def main():
    st.set_page_config(page_title="Clothing Classification and Recommendation", page_icon=":dress:", layout="wide")

    # Set custom CSS styles
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f0f0f0;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border-radius: 4px;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stImage {
            max-width: 300px;
            margin: 0 auto;
        }
        .stSuccess, .stInfo {
            font-size: 18px;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page:", ["Image Classification", "Real-time Classification"])

    if page == "Image Classification":
        image_classification_page()
    elif page == "Real-time Classification":
        real_time_classification_page()

    st.sidebar.markdown("---")
    st.sidebar.write("Developed by Mohammadsadegh Solouki, Bsc in Data Science, IU International University of Applied Sciences, Germany")
    st.sidebar.write("For questions or feedback, please contact mohamad-sadegh.solouki@iubh.de")

if __name__ == '__main__':
    main()