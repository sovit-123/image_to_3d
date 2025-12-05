FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Copy your files
COPY setup.sh /workspace/setup.sh
COPY image_to_texture.py /workspace/image_to_texture.py
COPY hunyuan3d_final_req.txt /workspace/hunyuan3d_final_req.txt
COPY birefnet_weights /workspace/birefnet_weights
COPY run.sh /workspace/run.sh

RUN chmod +x /workspace/setup.sh

# Create the HF cache directory
RUN mkdir -p /workspace/huggingface

# Create venv, install PyTorch 2.5.1, then run setup
RUN python3.10 -m venv /workspace/venv && \
    /bin/bash -c "export HF_HOME=/workspace/huggingface && \
    export TRANSFORMERS_CACHE=/workspace/huggingface/transformers && \
    export HF_DATASETS_CACHE=/workspace/huggingface/datasets && \
    source /workspace/venv/bin/activate && \
    pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118"

# Set environment variables for runtime
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface/transformers
ENV HF_DATASETS_CACHE=/workspace/huggingface/datasets

# Also add them to venv activation script so they're always set
RUN echo 'export HF_HOME=/workspace/huggingface' >> /workspace/venv/bin/activate && \
    echo 'export TRANSFORMERS_CACHE=/workspace/huggingface/transformers' >> /workspace/venv/bin/activate && \
    echo 'export HF_DATASETS_CACHE=/workspace/huggingface/datasets' >> /workspace/venv/bin/activate