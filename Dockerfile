FROM nvcr.io/nvidia/cuda:11.6.1-runtime-ubuntu20.04
WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install -y python3.8
RUN apt-get install -y python3-pip
RUN apt-get install -y git
RUN python3 --version
RUN pip3 --version
RUN apt-get install -y libz-dev libssl-dev libcurl4-gnutls-dev libexpat1-dev gettext cmake gcc
RUN pip3 install -r requirements.txt
RUN pip3 install ftfy regex tqdm
RUN pip3 install git+https://github.com/openai/CLIP.git

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
