FROM pytorch-base:latest
WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN mkdir uploads
RUN pip3 install -r requirements.txt
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
RUN apt-get update && apt-get -y install libz-dev libssl-dev libcurl4-gnutls-dev gettext libexpat1-dev cmake gcc
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# RUN pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip3 install -U sentence-transformers

RUN pip3 install git+https://github.com/openai/CLIP.git

COPY . .
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
