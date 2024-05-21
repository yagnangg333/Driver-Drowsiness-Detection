import cv2
import dlib
import imutils
import streamlit as st
from imutils import face_utils
from pygame import mixer
from scipy.spatial import distance


def initialize():
    mixer.init()
    mixer.music.load("music.wav")
    return mixer


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def Drowsiness_detection(cap, thresh, frame_check, mixer):
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

    flag = 0
    prev_state = None
    FRAME_WINDOW = st.image([])

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    cv2.drawContours(gray, [leftEyeHull], -1, (255, 0, 0), 1)
                    cv2.drawContours(gray, [rightEyeHull], -1, (255, 0, 0), 1)
                    cv2.putText(
                        gray,
                        "****************Drowsy!****************",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )
                    cv2.putText(
                        gray,
                        "****************Drowsy!****************",
                        (10, 325),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )
                    if mixer:
                        mixer.music.play()
            else:
                flag = 0
                cv2.drawContours(gray, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(gray, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.putText(
                    gray,
                    "****************Awake!****************",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    gray,
                    "****************Awake!****************",
                    (10, 325),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
        FRAME_WINDOW.image(gray)


def main():
    st.title("Drowsiness Detection")

    st.sidebar.title("Parameters")
    thresh = st.sidebar.slider(
        "Threshold", min_value=0.1, max_value=0.5, value=0.25, step=0.01
    )
    frame_check = st.sidebar.slider(
        "Frame Check", min_value=5, max_value=50, value=20, step=5
    )

    st.sidebar.title("Music")
    play_music = st.sidebar.checkbox("Play Alert Music")

    run = st.checkbox("Start Detection")

    if run:
        cap = cv2.VideoCapture(0)
        mixer = initialize() if play_music else None
        Drowsiness_detection(cap, thresh, frame_check, mixer)
    else:
        st.write("Stopped")


if __name__ == "__main__":
    main()
