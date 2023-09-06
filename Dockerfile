# *********************
FROM python:3.10.6-buster

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY api api
COPY models/yolov8/checkpoints/weights models/yolov8/checkpoints/weights
COPY src src
COPY app.py app.py
COPY yolov8n.pt yolov8n.pt
COPY setup.py setup.py
COPY Makefile Makefile

RUN pip install -e .
RUN apt-get update && apt-get install -y make
RUN apt-get install -y libopencv-dev
RUN apt-get install -y python3-opencv



# *********************
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
# *********************
