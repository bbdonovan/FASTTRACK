##############################################################
# Dockerfile for Ollama + LLaMA 3.2-3B-Instruct on GPU laptop
##############################################################
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set up PATH
ENV PATH="/root/.local/bin:$PATH"

# Install essentials and Python
RUN apt-get update && apt-get install -y \
    curl ca-certificates bash git wget build-essential \
	python3.10 python3.10-dev python3.10-distutils python3.10-venv \
&& ln -s /usr/bin/python3.10 /usr/bin/python \
&& curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
&& apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Ollama (Linux) – using their install script
RUN curl -fsSL https://ollama.com/install.sh | bash

# Install PyTorch (CUDA 12.1) and vLLM
#RUN pip install --upgrade pip && \
#    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121 && \
#    pip install vllm transformers bitsandbytes==0.46.1

# Workdir
WORKDIR /app

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Copy model folder with weights already locally downloaded
COPY ./models/Meta-Llama-3.2-3B-Instruct /models/Meta-Llama-3.2-3B-Instruct

# Copy the Ollama Modelfile
COPY Modelfile /app/Modelfile

# Expose the Ollama API port
EXPOSE 11434

# Default command
ENTRYPOINT ["/app/start.sh"]
