FROM nvidia/cuda:11.8.1-cudnn8-runtime-ubuntu20.

ENV PYTHON_VERSION=3.10

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    libcublas11 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python${PYTHON_VERSION} -m pip install --no-cache-dir -r requirements.txt

COPY remote_faster_whisper.py .

# Expose the port if needed
EXPOSE 5000

# CMD with configurable config.yaml path
CMD ["python3.10", "remote_faster_whisper.py", "-c", "/app/config.yaml"]
