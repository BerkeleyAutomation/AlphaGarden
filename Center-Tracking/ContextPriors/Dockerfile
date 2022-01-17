# FROM python:3.8.12-slim-buster
FROM paddlepaddle/paddle:2.2.0-gpu-cuda11.2-cudnn8

RUN apt-get update
# RUN apt-get install libgomp1
# RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY ./requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
WORKDIR /app/ContextPrior_Paddle

CMD python3 train.py