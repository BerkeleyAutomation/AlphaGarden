FROM python:3.7
ADD . /usr/src/app
WORKDIR /usr/src/app
EXPOSE 4000
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "index.py"]