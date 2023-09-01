def get_predictions(model, source, conf=0.25, iou=0.7, imgsz=640, classes=None, stream=False):
    """
    Performs object detection inference using the provided model and source.

    Args:
        model: The model to be used for inference.
        source: The source for performing inference (e.g., image path, video stream).
        conf (float): Confidence threshold for object detection.
        iou (float): Intersection over union threshold for non-maximum suppression.
        imgsz (int): Size of the input images.
        classes (list or None): List of classes to consider for object detection.
        stream (bool): If True, the source is a video stream.

    Returns:
        The results of object detection inference.
    """
    # Run inference on the provided source using the model and specified parameters
    results = model(source=source, conf=conf, iou=iou,
                    imgsz=imgsz, classes=classes, stream=stream)

    # Print a message indicating the completion of inference
    print("✅ Inference completed.")

    return results


def get_predicted_classes(model, results):
    """
    Retrieves the predicted classes from the results of object detection.
    Args:
        model: The model used for inference.
        results: The results of object detection inference.
    Returns:
        A list of predicted classes.
    """
    pred_class_list = []
    # Loop through the predicted class indices and retrieve the class names from the model's names list
    for c in results[0].boxes.cls:
        pred_class_list.append(model.names[int(c)])
    # Print a message indicating the completion of class retrieval
    print("✅ Predicted classes retrieved.")
    return pred_class_list


def get_predicted_shape(results):
    """
    Retrieves the original shape of the image from the results of object detection.
    Args:
        results: The results of object detection inference.
    Returns:
        The original shape of the image.
    """
    # Initialize img_shape variable to None
    img_shape = None
    # Loop through the results and update img_shape with the original shape of each image
    for r in results:
        img_shape = r.orig_shape
    # Print a message indicating the completion of shape retrieval
    print("✅ Original image shape retrieved.")
    return img_shape


def get_predicted_speed(results):
    """
    Retrieves the inference speed from the results of object detection.
    Args:
        results: The results of object detection inference.
    Returns:
        The inference speed.
    """
    # Initialize inf_speed variable to None
    inf_speed = None
    # Loop through the results and update inf_speed with the inference speed of each result
    for r in results:
        inf_speed = r.speed['inference']
    # Print a message indicating the completion of speed retrieval
    print("✅ Inference speed retrieved.")
    return inf_speed
