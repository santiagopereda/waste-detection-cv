import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import io
from PIL import Image
from src.models.yolo.predict_model import get_predictions
from torchvision.transforms import functional as F
import supervision as sv
from io import BytesIO
import pickle


def custom_clases(custom_classes, category_names):
    assigned_class_id = []
    if custom_classes:
        assigned_class = st.sidebar.multiselect(
            'Select Custom Classes', category_names, default=category_names[0])
        for each in assigned_class:
            assigned_class_id.append(category_names.index(each))
        if assigned_class_id is not None:
            return assigned_class_id
        return None


def get_image_from_serialized(serialized_data, dsize=(640, 640)):
    # Gets serialized data <BitesIO>
    bytes_data = serialized_data.getvalue()
    # Transforms serialized data to Image
    imageBGR = cv2.imdecode(np.frombuffer(
        bytes_data, np.uint8), cv2.IMREAD_ANYCOLOR)
    # Corrects Colors
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
    # Resizes the image
    resized_img = cv2.resize(imageRGB, dsize=dsize,
                             interpolation=cv2.INTER_AREA)
    return resized_img


def create_temporary_file(file_input, ext='.mp4', delete=False):
    tpfile = tempfile.NamedTemporaryFile(suffix=ext, delete=delete)
    tpfile.write(file_input.read())
    demo_binary = open(tpfile.name, 'rb')
    demo_bytes = demo_binary.read()
    return demo_bytes


def create_temp_video_from_byte_stream(video_bytes):
    # Create a temporary video file
    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    output_path = temp_video.name

    with open(output_path, 'wb') as f:
        f.write(video_bytes)

    return output_path


def annotate_image(results, frame, classes):
    detections = sv.Detections.from_yolov8(results[0])
    box_annotator = sv.BoxAnnotator()

    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for xyxy, confidence, class_id, _
        in detections
    ]

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections,
        labels=labels
    )

    return annotated_frame


def get_annotated_video(video, model, assigned_class_id, confidence):

    image_list = []
    cap = cv2.VideoCapture(video)

    classes = list(model.names.values())

    with st.spinner("Creating video..."):
        while cap.isOpened():

            # Read a frame from the video
            success, frame = cap.read()

            if success:

                results = get_predictions(
                    model, frame, classes=assigned_class_id, conf=confidence)

                annotated_frame = annotate_image(results, frame, classes)

                image_list.append(annotated_frame)

            else:
                # Break the loop if the end of the video is reached
                break

        cap.release()

        # Configure video parameters
        output_filename = '/home/spereda/code/santiagopereda/08-Project/waste-detection-cv/src/app/output_video.mp4'
        codec = cv2.VideoWriter_fourcc(*'MP4V')
        # Create VideoWriter object
        fps = 30
        frame_width, frame_height = image_list[0].shape[1], image_list[0].shape[0]

        out = cv2.VideoWriter(output_filename, codec, fps,
                              (frame_width, frame_height))

        # Write frames to the video
        for frame in image_list:
            out.write(frame)

        # Release the VideoWriter
        out.release()
