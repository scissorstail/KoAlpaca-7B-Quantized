FROM theeluwin/pytorch-ko:latest

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    make \
    git \
    git-lfs \
    curl \
    vim

COPY requirements.txt /workspace/

RUN pip install -r requirements.txt

