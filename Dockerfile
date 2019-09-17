FROM python:3

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:/app/keras-accent-trainer"

WORKDIR /app
RUN pip install --upgrade pip

ADD requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app