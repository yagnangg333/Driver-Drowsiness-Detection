"""Streamlit UI for driver drowsiness detection.

Run with:
    streamlit run Drowsiness_Detection_w_UI.py

The webcam is read with OpenCV on the machine running Streamlit, so this app
is meant to be run locally (not on a remote server / Streamlit Cloud).
"""

import cv2
import imutils
import streamlit as st

from detector import (
    DEFAULT_CONSEC_FRAMES,
    DEFAULT_EAR_THRESHOLD,
    FRAME_WIDTH,
    Alarm,
    DrowsinessDetector,
)


@st.cache_resource
def get_alarm() -> Alarm:
    return Alarm()


def run_detection(camera_index: int, detector: DrowsinessDetector, alarm) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error(f"Could not open camera {camera_index}. Is it in use by another app?")
        return

    frame_window = st.image([])
    status = st.empty()
    try:
        # Streamlit interrupts this loop (via an exception) when a widget
        # changes, so the finally block always releases the camera.
        while True:
            ok, frame = cap.read()
            if not ok:
                st.error("Failed to read a frame from the camera.")
                break

            frame = imutils.resize(frame, width=FRAME_WIDTH)
            result = detector.process(frame)

            if alarm and result.drowsy:
                alarm.start()
            elif alarm:
                alarm.stop()

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.face_found:
                status.warning("No face detected")
            elif result.drowsy:
                status.error(f"Drowsy! Wake up! (EAR {result.ear:.2f})")
            else:
                status.success(f"Awake (EAR {result.ear:.2f})")
    finally:
        if alarm:
            alarm.stop()
        cap.release()


def main() -> None:
    st.set_page_config(page_title="Drowsiness Detection", page_icon="😴")
    st.title("Driver Drowsiness Detection")

    st.sidebar.header("Parameters")
    threshold = st.sidebar.slider(
        "EAR threshold",
        min_value=0.10,
        max_value=0.50,
        value=DEFAULT_EAR_THRESHOLD,
        step=0.01,
        help="Eye aspect ratio below which the eyes are considered closed.",
    )
    consec_frames = st.sidebar.slider(
        "Consecutive frames",
        min_value=5,
        max_value=50,
        value=DEFAULT_CONSEC_FRAMES,
        step=1,
        help="How many closed-eye frames in a row trigger the alert.",
    )
    camera_index = st.sidebar.number_input("Camera index", min_value=0, value=0, step=1)

    st.sidebar.header("Alert")
    play_sound = st.sidebar.checkbox("Play alert sound", value=True)

    try:
        detector = DrowsinessDetector(threshold, consec_frames)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    alarm = get_alarm() if play_sound else None
    if alarm and not alarm.available:
        st.sidebar.warning("Audio unavailable; alert sound disabled.")
        alarm = None

    if st.toggle("Start detection"):
        run_detection(int(camera_index), detector, alarm)
    else:
        st.info("Detection stopped. Turn on **Start detection** to begin.")


if __name__ == "__main__":
    main()
