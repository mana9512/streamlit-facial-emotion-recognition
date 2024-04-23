import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# Load model 1
emotion_dict1 = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
json_file1 = open('cnn_model_latest.json', 'r')
loaded_model_json1 = json_file1.read()
json_file1.close()
classifier1 = model_from_json(loaded_model_json1)
classifier1.load_weights("cnn_model_latest.weights.h5")

# Load model 2
emotion_dict1 = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
json_file2 = open('restnet_model.json', 'r')
loaded_model_json2 = json_file2.read()
json_file2.close()
classifier2 = model_from_json(loaded_model_json2)
classifier2.load_weights("restnet_model.weights.h5")

# Load face cascade classifier
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class FaceEmotionModel1(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier1.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict1[maxindex]
                output = str(finalout)
                # Draw square around the face
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Display emotion label
                cv2.putText(img, output, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        return img


class FaceEmotionModel2(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier2.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict1[maxindex]
                output = str(finalout)
                # Draw square around the face
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Display emotion label
                cv2.putText(img, output, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        return img


def main():
    st.title("Facial Expression Detection Application")
    models = ["Home", "Custom CNN model", "ResNet50 based CNN model"]
    choice = st.sidebar.selectbox("Select Model", models)
    st.sidebar.markdown(
        """ Built by Manashree""")
    if choice == "Home":
        st.write("""
                 The application has two models to detect Facial Expression .
                 1. Custom CNN model.
                 2. ResNet50 based CNN model.
                 """)
    elif choice == "Custom CNN model":
        st.header("Custom CNN model")
        st.write("Click on start to use webcam and detect your face emotion with custom CNN model")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_transformer_factory=FaceEmotionModel1)
    elif choice == "ResNet50 based CNN model":
        st.header("ResNet50 based CNN model")
        st.write("Click on start to use webcam and detect your face emotion with ResNet50 based CNN model")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_transformer_factory=FaceEmotionModel2)
    else:
        pass


if __name__ == "__main__":
    main()
