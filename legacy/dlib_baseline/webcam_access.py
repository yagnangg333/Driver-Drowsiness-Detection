"""Quick Streamlit check that the webcam works.

Run with:
    streamlit run webcam_access.py
"""

import cv2
import streamlit as st

st.title("Webcam Live Feed")

if st.toggle("Run"):
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        st.error("Could not open camera 0.")
        st.stop()

    frame_window = st.image([])
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                st.error("Failed to read a frame from the camera.")
                break
            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        camera.release()
else:
    st.write("Stopped")
