# This container will build and run the Bad Meme Cropper
# Script.
FROM jjanzic/docker-python3-opencv
ADD . /src/
WORKDIR /src/

# Install python deps
RUN pip install arghelper

ENTRYPOINT ["python", "main.py"]
