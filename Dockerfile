FROM python:3.7-stretch

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN apt-get update -y
RUN apt-get upgrade -y

RUN apt-get install make && apt-get install -y software-properties-common && \
                            add-apt-repository main && \
                            apt-get install build-essential -y && \
                            apt-get install -y libsndfile1 python3-dev ffmpeg


WORKDIR /app
RUN pip install --upgrade pip

ADD requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8080