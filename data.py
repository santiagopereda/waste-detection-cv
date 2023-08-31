API_KEY = 'M5JOZygUKYjMDTlEyr5z'
PROJECT = 'lewagon'
VERSION = 1
DATA_LOCATION = '/home/vuyani/code/Santiago/waste-detection-cv/data/data-train'
from roboflow import Roboflow


def get_data(api_key,format,project,version,location):

    rf = Roboflow(api_key=api_key,model_format=format)
    dataset = rf.workspace().project(project).version(version).download(location=location)
    print(f"✅ Data saved in {DATA_LOCATION}")
    return dataset

if __name__ == '__main__':

    get_data(api_key=API_KEY,format='yolov8',
             project='lewagon',version=1,
             location=DATA_LOCATION)
