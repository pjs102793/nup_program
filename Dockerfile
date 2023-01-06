FROM nvcr.io/nvidia/tensorrt:22.04-py3

SHELL ["/bin/bash", "-c"]

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y locales libb64-dev libgl1-mesa-glx \
    && locale-gen ko_KR.UTF-8 \
    && pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 \
    && pip3 install tritonclient[all] opencv-python tqdm

ENV LC_ALL ko_KR.UTF-8

WORKDIR /workspace/test

RUN ["/bin/bash"]