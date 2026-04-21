# 👕 Style Match: Clothing Color Recommender

An interactive web application built with Python and Streamlit that analyzes a user's skin tone and undertone to suggest the most flattering clothing colors. 

This was developed as a college project to demonstrate the practical application of computer vision and machine learning techniques in fashion and personal styling.

## ✨ Features
* **Face Detection:** Utilizes OpenCV's Haar Cascade to accurately locate faces in uploaded images.
* **Color Extraction:** Applies K-Means clustering (`scikit-learn`) to extract the dominant skin color from targeted regions of the face.
* **Undertone Analysis:** A custom algorithm calculates RGB ratios to determine if the user has Warm, Cool, or Neutral undertones.
* **Smart Recommendations:** Provides a curated list of clothing colors to wear and colors to avoid based on established color theory.
* **Interactive UI:** Built entirely in Python using Streamlit for a seamless user experience.

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend/Framework:** Streamlit
* **Computer Vision:** OpenCV (`opencv-python-headless`)
* **Machine Learning:** Scikit-learn (K-Means Clustering)
* **Image Processing:** NumPy, Pillow

## 🚀 How to Run Locally
https://face-color-recognition-rihieseb.streamlit.app/
