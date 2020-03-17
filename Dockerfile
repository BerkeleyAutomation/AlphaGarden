FROM python:3.7
COPY . /app
WORKDIR /app
ADD ./RL_Framework/data_collection.py /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "./RL_Framework/data_collection.py"]
