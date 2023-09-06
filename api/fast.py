from typing import Annotated
from fastapi import FastAPI, File, Form
from fastapi.middleware.cors import CORSMiddleware
from src.models.yolo.train_model import define_model
from src.models.yolo.predict_model import get_predictions
from src.visualization.utils import get_image_from_buffer, get_encoded_image, create_temporary_file
app = FastAPI()


app.state.model = define_model('best')


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/classes")
async def models():
    return list(app.state.model.names.values())


@app.post("/predict_photo")
async def predict(
    resized_img: Annotated[bytes, File()],
    confidence: Annotated[str, Form()],
    assigned_class_id: Annotated[str, Form()],
):

    print(assigned_class_id)
    if assigned_class_id == "Empty":
        assigned_class_id = None
    else:
        assigned_class_id = [int(num) for num in assigned_class_id.split(',')]

    confidence = float(confidence)

    resized_img = get_image_from_buffer(resized_img)

    result = get_predictions(app.state.model, source=resized_img, conf=confidence,classes=assigned_class_id)

    for r in result:
        im_array = r.plot()

    img_base64 = get_encoded_image(im_array)

    return (img_base64)

@app.post("/predict_video")
async def predict(
    video: Annotated[bytes, File()],
    confidence: Annotated[str, Form()],
    assigned_class_id: Annotated[str, Form()],
):
    if assigned_class_id == 'a':
        assigned_class_id = None
    else:
        [int(num) for num in assigned_class_id.split(',')]
    confidence = float(confidence)

    demo_bytes = create_temporary_file(
                video, ext='.avi', delete=False)

    print(demo_bytes)

    msg = 'ok'

    return (msg)
