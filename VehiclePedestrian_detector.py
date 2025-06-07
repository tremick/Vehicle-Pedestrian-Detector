import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer


st.set_page_config(page_title="Vehicle & Pedestrian Detection", layout="centered")
st.title("🚗 Real-time Vehicle & Pedestrian Detection")
st.markdown("This app uses your webcam to detect **vehicles** and **pedestrians** in real-time using OpenCV's Haar cascades.")


vehicle_cascade_path = "haarcascade_car.xml"
pedestrian_cascade_path = "haarcascade_fullbody.xml"


veh_detector = cv2.CascadeClassifier(vehicle_cascade_path)
ped_detector = cv2.CascadeClassifier(pedestrian_cascade_path)


class VehiclePedestrianDetection(VideoTransformerBase):
    def transform(self, frame):
       
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

       
        vehicles = veh_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in vehicles:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Vehicle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

       
        pedestrians = ped_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Pedestrian", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return image


webrtc_streamer(
    key="vehicle-pedestrian-detection",
    video_transformer_factory=VehiclePedestrianDetection,
    media_stream_constraints={"video": True, "audio": False},
)
