# Use the official TensorFlow Docker image
FROM tensorflow/tensorflow:latest-gpu

COPY requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter and other Python packages
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

# Set the working directory
WORKDIR /workspace

# Install VSCode Server
# RUN curl -fsSL https://code-server.dev/install.sh | sh

# Command to start Jupyter Notebook
# CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root & code-server --bind-addr 0.0.0.0:8080 --auth none /workspace"]

# Expose Jupyter and VSCode Server ports
# EXPOSE 8888
# EXPOSE 8080