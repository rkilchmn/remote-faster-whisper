FROM ubuntu:22.04 AS base

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
    python3-pip
    
COPY requirements.txt .    

RUN python3 -m pip install --no-cache-dir -r requirements.txt

FROM base AS final

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    libsndfile1 \
    gnupg \
    curl \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends -y \
    nvidia-container-runtime \
    && rm -rf /var/lib/apt/lists/* 

ENV NVIDIA_DRIVER_CAPABILITIES=compute
ENV NVIDIA_VISIBLE_DEVICES=all

WORKDIR /app

COPY remote_faster_whisper.py .
COPY config.yaml .

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

# CMD with configurable config.yaml path
CMD ["python3", "remote_faster_whisper.py", "-c", "config.yaml"]
