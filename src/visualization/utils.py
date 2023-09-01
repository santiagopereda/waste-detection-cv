import os
import cv2
import glob
import random
import tempfile
import numpy as np
import streamlit as st
import supervision as sv
import matplotlib.pyplot as plt
from src.models.yolo.predict_model import get_predictions


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
        for xyxy, mask, confidence,  class_id,  tracker_id
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


def yolo2bbox(bboxes):
    """
    Converts YOLO-format bounding boxes to xmin, ymin, xmax, ymax format.
    Args:
        bboxes: YOLO-format bounding box coordinates (center_x, center_y, width, height).
    Returns:
        Tuple containing (xmin, ymin, xmax, ymax) bounding box coordinates.
    """
    # Calculate xmin, ymin, xmax, ymax based on YOLO-format bounding box coordinates
    xmin = bboxes[0] - bboxes[2] / 2
    ymin = bboxes[1] - bboxes[3] / 2
    xmax = bboxes[0] + bboxes[2] / 2
    ymax = bboxes[1] + bboxes[3] / 2

    return xmin, ymin, xmax, ymax


def plot_box(image, bboxes, labels):
    """
    Plots bounding boxes on the input image.
    Args:
        image: The input image on which to draw the bounding boxes.
        bboxes: List of bounding box coordinates in YOLO-format.
        labels: List of labels corresponding to each bounding box.
    Returns:
        The image with drawn bounding boxes.
    """
    # Get the image height and width to denormalize the bounding box coordinates
    h, w, _ = image.shape

    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)

        # Denormalize the coordinates
        xmin = int(x1 * w)
        ymin = int(y1 * h)
        xmax = int(x2 * w)
        ymax = int(y2 * h)

        # Determine the thickness of the bounding box based on the image width
        thickness = max(2, int(w / 275))

        cv2.rectangle(
            image,
            (xmin, ymin), (xmax, ymax),
            color=(0, 0, 255),
            thickness=thickness
        )

        # Add label text above the bounding box
        label = labels[box_num]
        cv2.putText(
            image, label,
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 255),
            thickness=1
        )

    return image


def plot(image_paths, label_paths, num_samples):
    """
    Plots images with bounding boxes using YOLO annotations.
    Args:
        image_paths: Path to the directory containing the images.
        label_paths: Path to the directory containing the YOLO label files.
        num_samples: Number of samples to display.
    Note:
        Ensure that the yolo2bbox and plot_box functions are defined and available.
    """
    all_images = []
    all_images.extend(glob.glob(os.path.join(image_paths, '*.jpg')))
    all_images.extend(glob.glob(os.path.join(image_paths, '*.JPG')))

    all_images.sort()

    num_images = len(all_images)

    plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        j = random.randint(0, num_images - 1)
        image_name = all_images[j]
        image_name = '.'.join(image_name.split(
            os.path.sep)[-1].split('.')[:-1])
        image = cv2.imread(all_images[j])
        with open(os.path.join(label_paths, image_name+'.txt'), 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label = label_line[0]
                bbox_string = label_line[2:]
                x_c, y_c, w, h = bbox_string.split(' ')
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_box(image, bboxes, labels)
        plt.subplot(2, 2, i+1)
        plt.imshow(result_image[:, :, ::-1])
        plt.axis('off')
    plt.subplots_adjust(wspace=0.5)  # Adjust spacing between subplots
    plt.tight_layout()
    plt.show()


def visualize(result_dir, num_samples=4):
    """
    Visualizes images from a directory.
    Args:
        result_dir: Path to the directory containing the images.
        num_samples: Number of samples to display.
    Note:
        Make sure the required images are present in the provided directory.
    """
    plt.figure(figsize=(20, 12))

    # Get a list of image file names in the result directory
    image_names = glob.glob(os.path.join(result_dir, '*.jpg'))

    # Shuffle the image names to get random samples
    random.shuffle(image_names)

    # Display the specified number of samples
    for i, image_name in enumerate(image_names):
        image = plt.imread(image_name)
        plt.subplot(2, 2, i+1)
        plt.imshow(image)
        plt.axis('off')
        if i == num_samples - 1:
            break

    plt.tight_layout()
    plt.show()
