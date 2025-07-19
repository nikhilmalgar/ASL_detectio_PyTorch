# Add path fix to import from parent directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import cv2
import numpy as np
import pyttsx3
import threading
import time
from torchvision import transforms
from src.model import ASLClassifier
import mediapipe as mp

# Setup TTS
engine = pyttsx3.init()

def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run).start()

# Load model
MODEL_PATH = os.path.join("saved_models", "asl_model.pth")
CLASS_NAMES = sorted(os.listdir("data/asl_alphabet_train"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLClassifier(num_classes=len(CLASS_NAMES)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

def get_hand_crop(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        h, w, _ = image.shape
        hand_landmarks = results.multi_hand_landmarks[0]

        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]

        x_min = int(min(x_coords) * w) - 30
        x_max = int(max(x_coords) * w) + 30
        y_min = int(min(y_coords) * h) - 30
        y_max = int(max(y_coords) * h) + 30

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        return image[y_min:y_max, x_min:x_max]
    return None

def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        prob = torch.nn.functional.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(prob).item()
        confidence = prob[pred_idx].item()
        return CLASS_NAMES[pred_idx], confidence

# Streamlit App
st.set_page_config(page_title="ASL Sign Recognizer", layout="centered")
st.title("🧠 ASL Hand Sign Recognition")
st.write("Enable webcam and show an ASL hand sign for prediction.")

run = st.checkbox("🎥 Enable Webcam")
frame_placeholder = st.empty()
prediction_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not access webcam.")
    else:
        prev_prediction = ""
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            hand_crop = get_hand_crop(frame)

            if hand_crop is not None:
                pred_class, confidence = predict(hand_crop)
                if pred_class != prev_prediction:
                    prediction_placeholder.subheader(f"🧾 Predicted: {pred_class} ({confidence:.2%})")
                    speak(pred_class)
                    prev_prediction = pred_class
            else:
                prediction_placeholder.subheader("🖐 No hand detected...")

            frame_display = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (300, 300))
            frame_placeholder.image(frame_display, channels="RGB")

            time.sleep(0.15)

        cap.release()
        cv2.destroyAllWindows()
