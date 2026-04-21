
import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# --- FUNCTIONS ---
def detect_undertone(r, g, b):
    """Approximates skin undertone based on RGB ratios."""
    if r > b + 30:
        return "Warm"
    elif r > b + 15:
        return "Neutral"
    else:
        return "Cool"

def classify_skin_tone(rgb):
    """Classifies skin tone based on luminance."""
    r, g, b = rgb
    luminance = 0.299*r + 0.587*g + 0.114*b
    if luminance > 200: return "Fair"
    elif luminance > 170: return "Light"
    elif luminance > 140: return "Medium"
    elif luminance > 110: return "Olive"
    elif luminance > 80: return "Brown"
    else: return "Deep"

# Dictionary for clothing recommendations
recommendations = {
    "Warm": {
        "best": ["Earthy Reds", "Mustard Yellow", "Olive Green", "Warm Brown", "Peach"],
        "avoid": ["Icy Blues", "Jewel Tones"]
    },
    "Cool": {
        "best": ["Navy Blue", "Emerald Green", "Deep Purple", "Bright White", "Icy Pink"],
        "avoid": ["Orange", "Mustard", "Tomato Red"]
    },
    "Neutral": {
        "best": ["Dusty Pink", "Jade Green", "Cornsilk Yellow", "Mid-level Grays"],
        "avoid": ["Overly bright neon colors"]
    }
}

# --- STREAMLIT UI ---
st.title("Style Match: Clothing Color Recommender 👕")
st.write("Upload a front-facing photo to find the best clothing colors for your skin tone.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption='Uploaded Image', use_container_width=True)

    # Face Detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.error("No face detected! Try a clearer front-facing photo.")
    else:
        st.success("Face detected successfully!")
        x, y, w, h = faces[0]

        # Draw bounding box
        img_copy = img_rgb.copy()
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)
        st.image(img_copy, caption='Face Detected', use_container_width=True)

        # Extract skin region (forehead/cheeks)
        skin_region = img_rgb[y:y+int(h*0.4), x+int(w*0.2):x+int(w*0.8)]
        pixels = skin_region.reshape(-1, 3).astype(float)
        
        # Calculate dominant color
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pixels)
        dominant_colour = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))].astype(int)

        tone = classify_skin_tone(dominant_colour)
        undertone = detect_undertone(*dominant_colour)

        st.write("---")
        st.header("Your Results")
        
        # Display the color block
        hex_color = '#{:02x}{:02x}{:02x}'.format(dominant_colour[0], dominant_colour[1], dominant_colour[2])
        st.write("**Dominant Skin Color:**")
        st.markdown(f'<div style="background-color: {hex_color}; width: 100%; height: 50px; border-radius: 5px; border: 1px solid #ccc;"></div>', unsafe_allow_html=True)
        
        st.write(f"- **Skin Tone Category:** {tone}")
        st.write(f"- **Skin Undertone:** {undertone}")

        st.write("---")
        st.header("Wardrobe Recommendations 👗👔")
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("**Best Colors for You:**")
            for color in recommendations[undertone]["best"]:
                st.write(f"✅ {color}")
        
        with col2:
            st.error("**Colors to Avoid:**")
            for color in recommendations[undertone]["avoid"]:
                st.write(f"❌ {color}")
