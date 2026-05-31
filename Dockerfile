FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY datasets/ ./datasets/
COPY base_model/ ./base_model/
COPY finetuned_model/ ./finetuned_model/

# Default: drop into an interactive shell so you can run scripts manually
# Override by appending a command, e.g:
#   docker run --gpus all -it ghcr.io/tumisomokautu/zulu-classifier:latest python3 scripts/finetune.py
#   docker run --gpus all -it ghcr.io/tumisomokautu/zulu-classifier:latest python3 scripts/test.py
CMD ["bash"]
