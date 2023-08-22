FROM python:3.11.3-slim

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["GestureWithImage.py" ]