import cv2

import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
veh_detector = cv2.CascadeClassifier("haarcascade_car.xml")
ped_detector = cv2.CascadeClassifier("haarcascade_fullbody.xml")
class FaceDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        vehicle = veh_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in vehicle:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        
        pedestrian = ped_detector.detectMultiScale(gray)
        for (ex, ey, ew, eh) in pedestrian:
            cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        return image
    

    st.set_page_config(page_title="Live Face Detection", layout="centered")
st.title("📸 Real-time Face & Eye Detection")

st.markdown("This app uses your webcam to detect faces and eyes in real-time.")
webrtc_streamer(
    key="face-detection",
    video_transformer_factory=FaceDetectionTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

