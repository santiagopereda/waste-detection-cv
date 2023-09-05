import base64
import requests
import streamlit as st
from src.params_yolo import *
from src.visualization.utils import custom_clases, create_temporary_file


def use_requests(api_url, params):
    response = requests.get(api_url, params=params)
    return response

def main():

    predict_url = 'http://127.0.0.1:8000/predict_photo'
    
    predict_video_url = 'http://127.0.0.1:8000/predict_video'

    supported_image_list = ['bmp', 'dng', 'jpeg',
                            'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm']

    supported_video_list = ['asf', 'avi', 'gif', 'm4v',
                            'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm']

    category_names = requests.get("http://127.0.0.1:8000/classes").json()

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
                
                if assigned_class_id == None:
                    assigned_class_id = "a"
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
                
                st.image(base64.b64decode((response.text)))

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
                
                if assigned_class_id == None:
                    assigned_class_id = "a"
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
  
                st.image(base64.b64decode((response.text)))


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
