# Deploying RedHat's FP8 Gemma-3 27B with vLLM for High-Performance Inference

This guide provides a complete, step-by-step walkthrough for deploying the `RedHatAI/gemma-3-27b-it-FP8-dynamic` model using the vLLM inference server. The result is a secure, high-performance, OpenAI-compatible API endpoint capable of text, multi-language, and multi-modal (image-text) inference.

The instructions cover everything from system setup to launching the server and verifying its functionality with a suite of Python test scripts.

## Table of Contents
1.  [Prerequisites](#1-prerequisites)
2.  [Environment Setup](#2-environment-setup)
    - 2.1 Install Docker
    - 2.2 Install NVIDIA Container Toolkit
3.  [Model and Server Preparation](#3-model-and-server-preparation)
    - 3.1 Pull the vLLM Docker Image
    - 3.2 Download the Gemma-3 FP8 Model
4.  [Launch the Inference Server](#4-launch-the-inference-server)
5.  [Testing the API Endpoint](#5-testing-the-api-endpoint)
    - 5.1 Install Python Dependencies
    - 5.2 Prepare the Test Image
    - 5.3 Run Test Scripts
6.  [Managing the Server](#6-managing-the-server)

---

## 1. Prerequisites

-   An NVIDIA GPU with at least 48 GB of VRAM (e.g., L40S, A100) is recommended for the specified `max-model-len`.
-   A Linux-based OS (this guide uses Ubuntu commands).
-   NVIDIA drivers installed on the host machine.
-   A Hugging Face account and an access token with permissions to download models.

## 2. Environment Setup

### 2.1 Install Docker
First, set up Docker's official repository and install the Docker Engine.

```bash
# Set up Docker's official repository
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install the modern Docker Engine, CLI, and Compose plugin
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 2.2 Install NVIDIA Container Toolkit
This toolkit allows Docker containers to access your NVIDIA GPU.

```bash
# Add the NVIDIA repository and key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime and restart the Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 3. Model and Server Preparation

### 3.1 Pull the vLLM Docker Image
Pull the latest official vLLM image, which includes an OpenAI-compatible server.

```bash
docker pull vllm/vllm-openai:latest
```

### 3.2 Download the Gemma-3 FP8 Model
Log in to Hugging Face and download the model files to your local cache.

```bash
# Install the Hugging Face command-line tool
pip install huggingface_hub

# Log in with your Hugging Face token (it will prompt you to paste it)
huggingface-cli login

# Download the model files. They will be stored in ~/.cache/huggingface/
# We use the repository ID directly; no need for --local-dir
huggingface-cli download RedHatAI/gemma-3-27b-it-FP8-dynamic
```

## 4. Launch the Inference Server

The following command starts the vLLM server in a detached container, mounts your Hugging Face cache, and secures the endpoint with an API key.

> **IMPORTANT:** Replace `YOUR_SUPER_SECRET_KEY` with your own secure key. This key will be required to authenticate all API requests.

```bash
docker run -d --gpus all --name gemma3-server -p 24434:8000 \
-v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
vllm/vllm-openai:latest \
--model RedHatAI/gemma-3-27b-it-FP8-dynamic \
--host 0.0.0.0 \
--port 8000 \
--max-model-len 131072 \
--gpu-memory-utilization 0.95 \
--trust-remote-code \
--api-key YOUR_SUPER_SECRET_KEY```

After running the command, monitor the server's startup process to ensure it loads correctly.

```bash
docker logs -f gemma3-server
```
Wait until you see the message `Application startup complete.` before proceeding. You can exit the logs with `Ctrl+C`.

## 5. Testing the API Endpoint

This section provides four Python scripts to validate different server capabilities.

### 5.1 Install Python Dependencies
Install the `openai` library to interact with the API and `Pillow` for image handling.

```bash
pip install openai pillow requests
```

### 5.2 Prepare the Test Image
Download a sample image that will be used for the multi-modal test script.

```bash
curl -o test_image.jpg "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
```
This will save `test_image.jpg` in your current directory.

### 5.3 Run Test Scripts

For each script below, **you must replace `YOUR_SUPER_SECRET_KEY` with the same key you used to launch the server.**In the tests folder there are 4 tests
1. test_english.py : test english streaming 
2. test_bangla.py : test bangla streaming 
3. test_image.py : test with an image 
4. test_concurrency.py : test concurrent requests

## 6. Managing the Server

To stop and remove the container when you are finished, use the following commands:

```bash
# Stop the background container
docker stop gemma3-server

# Remove the container
docker rm gemma3-server
```


# Appendix 

## For controlled faster service of text only services

```bash
docker run -d --gpus all --name gemma3-server -p 24434:8000 \
-v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
vllm/vllm-openai:latest \
--model RedHatAI/gemma-3-27b-it-FP8-dynamic \
--host 0.0.0.0 \
--port 8000 \
--max-model-len 32768 \
--gpu-memory-utilization 0.9 \
--max-num-batched-tokens 8192 \
--trust-remote-code \
--api-key YOUR_SUPER_SECRET_KEY \
--enable-chunked-prefill \
--max-num-seqs 256 \
--kv-cache-dtype fp8 \
--enforce-eager
```
