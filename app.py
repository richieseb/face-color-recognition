import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import mediapipe as mp
import math

# --- INITIALIZE MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# --- FUNCTIONS ---
def detect_undertone(r, g, b):
    """Approximates skin undertone based on RGB ratios."""
    if r > b + 30: return "Warm"
    elif r > b + 15: return "Neutral"
    else: return "Cool"

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

def calculate_distance(p1, p2, w, h):
    """Calculates true pixel distance between two normalized points."""
    return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)

def detect_face_shape(landmarks, w, h):
    """Calculates facial proportions to estimate face shape."""
    top_head = landmarks[10]
    bottom_chin = landmarks[152]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    left_jaw = landmarks[132]
    right_jaw = landmarks[361]

    # Calculate actual pixel distances
    face_length = calculate_distance(top_head, bottom_chin, w, h)
    face_width = calculate_distance(left_cheek, right_cheek, w, h)
    jaw_width = calculate_distance(left_jaw, right_jaw, w, h)

    # Heuristics for face shape
    if face_length > face_width * 1.2:
        if jaw_width > face_width * 0.8:
            return "Rectangular"
        else:
            return "Oval"
    elif face_width > jaw_width * 1.25:
        return "Heart"
    else:
        if jaw_width > face_width * 0.85:
            return "Square"
        else:
            return "Round"

# --- DICTIONARIES ---
color_recommendations = {
    "Warm": {"best": ["Earthy Reds", "Mustard Yellow", "Olive Green", "Warm Brown", "Peach"], "avoid": ["Icy Blues", "Jewel Tones"]},
    "Cool": {"best": ["Navy Blue", "Emerald Green", "Deep Purple", "Bright White", "Icy Pink"], "avoid": ["Orange", "Mustard", "Tomato Red"]},
    "Neutral": {"best": ["Dusty Pink", "Jade Green", "Cornsilk Yellow", "Mid-level Grays"], "avoid": ["Overly bright neon colors"]}
}

glasses_recommendations = {
    "Oval": "Aviators, Wayfarers, and virtually any frame shape.",
    "Round": "Rectangular, Square, and Angular frames to add structure.",
    "Square": "Round, Oval, and Cat-eye frames to soften the jawline.",
    "Heart": "Bottom-heavy frames, Rimless, or Aviators to balance the forehead.",
    "Rectangular": "Oversized, Tall, and Thick frames to break up face length."
}

# --- STREAMLIT UI ---
st.set_page_config(page_title="Style Match", layout="centered")
st.title("Style Match: Total Makeover 👓👕")
st.write("Upload a front-facing photo to find your best clothing colors and glasses shapes.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    st.image(img_rgb, caption='Uploaded Image', use_container_width=True)

    with st.spinner("Analyzing facial features and skin tone..."):
        # Process with MediaPipe
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            st.error("No face detected! Try a clearer front-facing photo without hair or shadows covering your face.")
        else:
            st.success("Face successfully mapped!")
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 1. FACE SHAPE ANALYSIS
            face_shape = detect_face_shape(landmarks, w, h)

            # 2. SKIN TONE ANALYSIS (Calculate bounding box from landmarks)
            x_coords = [landmark.x for landmark in landmarks]
            y_coords = [landmark.y for landmark in landmarks]
            
            # Convert to pixel coordinates and ensure they are within image bounds
            x_min, x_max = max(0, int(min(x_coords)*w)), min(w, int(max(x_coords)*w))
            y_min, y_max = max(0, int(min(y_coords)*h)), min(h, int(max(y_coords)*h))
            
            face_w = x_max - x_min
            face_h = y_max - y_min

            # Crop forehead/cheek area (top 40% of the face)
            skin_region = img_rgb[y_min : y_min+int(face_h*0.4), x_min+int(face_w*0.2) : x_min+int(face_w*0.8)]
            
            # K-Means Clustering
            pixels = skin_region.reshape(-1, 3).astype(float)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_colour = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))].astype(int)

            tone = classify_skin_tone(dominant_colour)
            undertone = detect_undertone(*dominant_colour)

            # --- DISPLAY RESULTS ---
            st.markdown("---")
            st.header("✨ Your Style Profile")
            
            # Color Output
            hex_color = '#{:02x}{:02x}{:02x}'.format(dominant_colour[0], dominant_colour[1], dominant_colour[2])
            st.write("**Dominant Skin Color:**")
            st.markdown(f'<div style="background-color: {hex_color}; width: 100%; height: 50px; border-radius: 5px; border: 1px solid #ccc;"></div><br>', unsafe_allow_html=True)
            
            # Profile Details
            st.write(f"🧔 **Face Shape:** {face_shape}")
            st.write(f"🎨 **Skin Tone Category:** {tone}")
            st.write(f"🌡️ **Skin Undertone:** {undertone}")

            st.markdown("---")
            
            # Recommendations Side-by-Side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👔 Clothing Colors")
                st.success("**Colors to Wear:**\n" + "\n".join([f"- {c}" for c in color_recommendations[undertone]["best"]]))
                st.error("**Colors to Avoid:**\n" + "\n".join([f"- {c}" for c in color_recommendations[undertone]["avoid"]]))

            with col2:
                st.subheader("👓 Glasses Frames")
                st.info(f"**Best Frames for {face_shape} Face:**\n\n{glasses_recommendations[face_shape]}")
