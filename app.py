import base64
import time
import requests
import streamlit as st
from src.params_yolo import *
from src.visualization.utils import custom_clases, create_temporary_file


def use_requests(api_url, params):
    response = requests.get(api_url, params=params)
    return response


def main():

    predict_url = f'{SERVICE_URL}/predict_photo'

    predict_video_url = f'{SERVICE_URL}/predict_video'

    supported_image_list = ['bmp', 'dng', 'jpeg',
                            'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm']

    supported_video_list = ['asf', 'avi', 'gif', 'm4v',
                            'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm']

    category_names = requests.get(f"{SERVICE_URL}/classes").json()

    # Empieza el flow

    # st.title('Waste Detection')

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

    menu = ["Image", "Live", "About"]

    st.sidebar.subheader("Parameters")

    # Todos los submenus
    choice = st.sidebar.selectbox("Choose App Mode", menu)

    st.sidebar.markdown('---')

    if choice == "Home":
        pass

    elif choice == "Image":

        st.subheader("**Deep learning-based waste detection model**")

        st.markdown("Waste detection and classification")

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

            if assigned_class_id == None or assigned_class_id == []:
                assigned_class_id = "Empty"
                image_dict = {
                    "resized_img": base64.b64encode(data_file.getvalue()).decode('utf-8'),
                    "confidence": str(confidence),
                    "assigned_class_id": assigned_class_id
                }

            else:
                image_dict = {
                    "resized_img": base64.b64encode(data_file.getvalue()).decode('utf-8'),
                    "confidence": str(confidence),
                    "assigned_class_id": ",".join(str(e) for e in assigned_class_id)
                }

            response = requests.post(predict_url, data=image_dict)

            if st.sidebar.button("Test Model"):

                placeholder = st.empty()

                progress_bar = st.progress(0)
                for perc_completed in range(100):
                    time.sleep(0.03)
                    progress_bar.progress(perc_completed+1)

                if assigned_class_id == 'a':
                    st.success(f"Photo uploaded succesfully!")
                else:
                    st.success(f"Photo uploaded succesfully!")

                with placeholder.container():
                    with st.spinner('Wait for it...'):
                        time.sleep(3)
                    st.image(base64.b64decode((response.text)))

    elif choice == "Video":

        custom_classes = st.sidebar.checkbox("Use Custom Classes")

        assigned_class_id = custom_clases(custom_classes, category_names)

        st.sidebar.markdown('---')

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

            if assigned_class_id == None:
                assigned_class_id = "a"
                video_dict = {
                    "video": base64.b64encode(data_file.getvalue()).decode('utf-8'),
                    "confidence": str(confidence),
                    "assigned_class_id": assigned_class_id
                }
            else:
                video_dict = {
                    "video": base64.b64encode(data_file.getvalue()).decode('utf-8'),
                    "confidence": str(confidence),
                    "assigned_class_id": ",".join(str(e) for e in assigned_class_id)
                }

            if st.sidebar.button("Test Model"):

                response = requests.post(predict_video_url, data=video_dict)

    elif choice == "Live":
        st.subheader("Live Action!")

        custom_classes = st.sidebar.checkbox("Use Custom Classes")

        assigned_class_id = custom_clases(custom_classes, category_names)

        st.sidebar.markdown('---')

        confidence = st.sidebar.slider(
            'Confidence', min_value=0.0, max_value=1.0, value=0.25)

        st.sidebar.markdown('---')

        data_file = st.camera_input("Take a picture")

        if data_file is not None:

            if assigned_class_id == None:
                assigned_class_id = "Empty"
                image_dict = {
                    "resized_img": base64.b64encode(data_file.getvalue()).decode('utf-8'),
                    "confidence": str(confidence),
                    "assigned_class_id": assigned_class_id
                }
            else:
                image_dict = {
                    "resized_img": base64.b64encode(data_file.getvalue()).decode('utf-8'),
                    "confidence": str(confidence),
                    "assigned_class_id": ",".join(str(e) for e in assigned_class_id)
                }

            response = requests.post(predict_url, data=image_dict)

            placeholder = st.empty()

            progress_bar = st.progress(0)
            for perc_completed in range(100):
                time.sleep(0.03)
                progress_bar.progress(perc_completed+1)

            if assigned_class_id == 'a':
                st.success(f"Photo uploaded succesfully!")
            else:
                st.success(f"Photo uploaded succesfully!")

            with placeholder.container():
                with st.spinner('Wait for it...'):
                    time.sleep(3)

            st.write('Drumroolllls!!!')
            with st.expander('Click to see the result!'):

                st.image(base64.b64decode((response.text)))

    elif choice == 'About':
        st.subheader("**Deep learning-based waste detection model**")

        st.subheader("**The Team**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("Gurban Abbasov")
            st.image('ga.jpg')
            st.markdown("https://github.com/Gurban1990")

        with col2:
            st.markdown("Santiago Pereda")
            st.image('sp.jpg')
            st.markdown("https://github.com/santiagopereda")

        with col3:
            st.markdown("Vuyani Jaka")
            st.image('vj.jpg')
            st.markdown("https://github.com/Vuyani6")


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
