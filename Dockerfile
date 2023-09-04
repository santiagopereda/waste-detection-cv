FROM python:3.10.6-buster
COPY src src
COPY app.py app.py
COPY ./models/yolov8/yolov8n.pt ./models/yolov8/yolov8n.pt
COPY requirements.txt requirements.txt
COPY Makefile Makefile
RUN apt-get update && apt-get install libgl1 python3-opencv
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install .
RUN make reinstall_package
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]