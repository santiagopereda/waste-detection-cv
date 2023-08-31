import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from io import BytesIO
from PIL import Image
from src.models.yolo.train_model import define_model
from src.models.yolo.predict_model import get_predictions
from src.app.utils import custom_clases, get_image_from_serialized, create_temporary_file, get_annotated_video, create_temp_video_from_byte_stream


def main():

    # Here i declare all my functions and variables

    model = define_model()

    supported_image_list = ['bmp', 'dng', 'jpeg',
                            'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm']

    supported_video_list = ['asf', 'avi', 'gif', 'm4v',
                            'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm']

    category_names = list(model.names.values())

    # Empieza el flow

    st.title('Waste Detection')

    st.sidebar.title('Settings')

    st.markdown(
        """
    <style>
    [data-testid='stSidebar'][aria-expanded='true'] > div:first-child{width: 300px;}
    [data-testid='stSidebar'][aria-expanded='false'] > div:first-child{width: 300px; margin-left:-480px}
    </style>
    """,
        unsafe_allow_html=True,
    )

    menu = ["Home", "Image", "Video", "Live", "About"]

    st.sidebar.subheader("Parameters")

    # Todos los submenus
    choice = st.sidebar.selectbox("Choose App Mode", menu)

    st.sidebar.markdown('---')

    if choice == "Home":
        pass

    elif choice == "Image":

        # Using custom classes for the prediction Bool
        custom_classes = st.sidebar.checkbox("Use Custom Classes")

        # Selects the categories to pass to be predicted
        assigned_class_id = custom_clases(custom_classes, category_names)

        st.sidebar.markdown('---')

        # Minimun confidence level to be predicted <float>
        confidence = st.sidebar.slider(
            'Confidence', min_value=0.0, max_value=1.0, value=0.25)

        st.sidebar.markdown('---')

        # Image Uploader
        data_file = st.sidebar.file_uploader(
            "Upload Image", type=supported_image_list)

        if data_file is not None:

            st.sidebar.text('Input Image')

            st.sidebar.image(data_file)

            if st.sidebar.button("Test Model"):

                file_details = {"Filename": data_file.name,
                                "FileType": data_file.type, "FileSize": data_file.size}

                resized_img = get_image_from_serialized(
                    data_file, dsize=(640, 640))

                st.image(resized_img)

                st.write(file_details)

                result = get_predictions(
                    model, resized_img, classes=assigned_class_id, conf=confidence)

                for r in result:
                    im_array = r.plot()  # plot a BGR numpy array of predictions

                    st.image(im_array, caption='Applied Model')

    elif choice == "Video":
        # Try Enable GPU on the prediction <bool> """TEST"""
        # enable_gpu = st.sidebar.checkbox('Enable GPU')

        # Using custom classes to be predicted <bool>
        custom_classes = st.sidebar.checkbox("Use Custom Classes")

        # Selects the categories to pass to be predicted
        assigned_class_id = custom_clases(custom_classes, category_names)

        st.sidebar.markdown('---')

        # Minimun confidence level to be predicted <float>
        confidence = st.sidebar.slider(
            'Confidence', min_value=0.0, max_value=1.0, value=0.25)

        st.sidebar.markdown('---')

        data_file = st.sidebar.file_uploader(
            "Upload Video", type=supported_video_list)

        if data_file:
            demo_bytes = create_temporary_file(
                data_file, ext='.avi', delete=False)

            st.sidebar.text('Input Video')

            st.video(demo_bytes)

            # st.write(type(demo_bytes))

            if st.sidebar.button("Test Model"):

                temp_video_path = create_temp_video_from_byte_stream(
                    demo_bytes)

                get_annotated_video(
                    temp_video_path, model, assigned_class_id=assigned_class_id, confidence=confidence)

                video_path = '/home/spereda/code/santiagopereda/08-Project/waste-detection-cv/src/app/'
                for filename in os.listdir(video_path):
                    if (filename.endswith(".mp4")): #or .avi, .mpeg, whatever.
                        os.system("ffmpeg -i {0} -c:v copy -c:a copy bunny.mp4".format(filename))
                    else:
                        continue
                          
                video_file = open('/home/spereda/code/santiagopereda/08-Project/waste-detection-cv/src/app/output_video.mp4', 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)

                



                # cst.video(temp_video_path)

    elif choice == "Live":
        st.subheader("Live Action!")

        # Using custom classes to be predicted <bool>
        custom_classes = st.sidebar.checkbox("Use Custom Classes")

        # Selects the categories to pass to be predicted
        assigned_class_id = custom_clases(custom_classes, category_names)

        st.sidebar.markdown('---')

        # Minimun confidence level to be predicted <float>
        confidence = st.sidebar.slider(
            'Confidence', min_value=0.0, max_value=1.0, value=0.25)

        st.sidebar.markdown('---')

        data_file = st.camera_input("Take a picture")

        if data_file is not None:
            with st.expander('Click to see the result!'):
                st.write('Drumroolllls!!!')
                resized_img = get_image_from_serialized(
                    data_file, dsize=(640, 640))

                result = get_predictions(
                    model, resized_img, classes=assigned_class_id, conf=confidence)

                for r in result:
                    im_array = r.plot()  # plot a BGR numpy array of predictions

                    st.image(im_array, caption='Applied Model')


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
