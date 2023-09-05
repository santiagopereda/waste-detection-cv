FROM python:3.10.6-buster
COPY src src
COPY app.py app.py
COPY ./models/yolov8/yolov8n.pt ./models/yolov8/yolov8n.pt
COPY requirements.txt requirements.txt
COPY Makefile Makefile
RUN apt-get update && apt-get install -y make
RUN apt-get install -y libopencv-dev 
RUN apt-get install -y python3-opencv
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
#CMD uvicorn api.simple:app --host 0.0.0.0 --port $PORT