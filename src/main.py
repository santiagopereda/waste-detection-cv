from src.data import *
from src.params_yolo import *
from src.data.make_dataset import load_data, save_model_gcp
from src.models.yolo.train_model import define_model, train_model



def master_yolo():

    load_data(key=API_KEY, project=WORKSPACE_PROJECT, version=WORKSPACE_PROJECT_VERSION, data_type=DATA_TYPE)
    print("✅ Data loading is completed")
    
    model = define_model(model_target=MODEL_TARGET)
    print("🚀 Model is ready to train")

    results = train_model(model=model, epochs=1)
    print('🏁 Model is now ready')
    if MODEL_SAVE == 'gcp':
        save_model_gcp()
    
    
master_yolo()









    
    
