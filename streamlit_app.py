import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import face_recognition
import os
import time
import cv2 
import keras
import numpy as np
import pandas as pd
import BodyPoseService as bps
import FacialService as fs
st.title("Image processing demo")

@st.cache_resource
def load_model():
    return YOLO("yolo11s.pt")


known_encodings = []
known_names = []
detect_model = load_model()

def load_student_encodings(folder="Student"):
    known_encodings, known_names = [], []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = face_recognition.load_image_file(path)
        face_zone = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, face_zone)
        if encodings:
            known_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_names.append(name)
    return known_encodings, known_names

known_encodings, known_names = load_student_encodings()

def remove_nested_boxes(boxes, threshold=0.8):
    """
    boxes: list of (x1, y1, x2, y2, conf, cls)
    threshold: fraction of area overlap to consider nested
    """
    keep = []
    for i, boxA in enumerate(boxes):
        x1A, y1A, x2A, y2A, confA, clsA, distanceA = boxA
        areaA = (x2A - x1A) * (y2A - y1A)
        nested = False

        for j, boxB in enumerate(boxes):
            if i == j:
                continue
            x1B, y1B, x2B, y2B, confB, clsB, distanceB = boxB

            # Intersection
            inter_x1 = max(x1A, x1B)
            inter_y1 = max(y1A, y1B)
            inter_x2 = min(x2A, x2B)
            inter_y2 = min(y2A, y2B)

            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                # fraction of A inside B
                frac_inside = inter_area / areaA
                if frac_inside >= threshold:
                    nested = True
                    break

        if not nested:
            keep.append(boxA)
    return keep

facialModel = fs.FacialService()
bodyPoseModel = bps.BodyPoseService()

def handle_facial_body_process(person_crop,model_facial,model_body):
    resultFacial = model_facial.extractionFactial(person_crop)
    resultPoseBody = model_body.extractionBodyPose(person_crop)
    combined = pd.concat([resultPoseBody,resultFacial], axis=1)
    return resultPoseBody
#1 Get the files
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 2. Read as PIL image 
    input_image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Original image")
    st.image(input_image)

    # 3. Process image
    img_array = np.array(input_image)
    results = detect_model.predict(
        source=img_array,
        imgsz=640,
        conf=0.25,
        iou=0.65,
        max_det=300,
        classes=[0], 
        device=''
    )
    result = results[0]
    boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        person_crop = img_array[y1:y2, x1:x2]
        #feature extraction
        features = handle_facial_body_process(person_crop,facialModel,bodyPoseModel)
        print(features)
        #face recognition
        face_locations = face_recognition.face_locations(person_crop)
        face_encodings = face_recognition.face_encodings(person_crop,face_locations)
        name = "Unknown"
        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(known_encodings,face_encoding)
            best_match_index = np.argmin(face_distances)
            if (face_distances[best_match_index] and face_distances[best_match_index] < 0.25):
                name = known_names[best_match_index]
        boxes.append((x1, y1, x2, y2, conf, name, face_distances[best_match_index]))
    filtered_boxes = remove_nested_boxes(boxes, threshold=0.9)
    #draw
    processed_array = img_array.copy()
    for (x1, y1, x2, y2, conf, name, distance) in filtered_boxes:
        cv2.rectangle(processed_array, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(processed_array, f"{name}", (x1, max(y1-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    #result
    processed_image = Image.fromarray(processed_array)

    # 4. Show processed image
    st.subheader("Processed image")
    st.image(processed_image)