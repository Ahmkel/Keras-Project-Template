FROM python:3

WORKDIR /app
RUN pip install --upgrade pip


ADD requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app