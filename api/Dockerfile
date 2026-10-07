# Same Python version as the local venv (3.13)
FROM python:3.13-slim

WORKDIR /app

# 1) CPU-only torch first (much smaller than the default CUDA build).
#    requirements.txt pins torch==2.14.1, which this already satisfies.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

# 2) The other dependencies. Copied before main.py so this slow layer is cached
#    and only rebuilds when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) The code. The model weights are NOT copied in; they are downloaded at runtime.
COPY main.py .

# Where the model comes from: a Hugging Face model repo (downloaded on first start).
# To use a local folder instead, mount it and override:
#   docker run -v <folder>:/model -e CHECKPOINT_PATH=/model ...
ENV CHECKPOINT_PATH=HamidAmeer/hassaniya-whisper-medium

# Writable cache folder for downloaded weights (hosts like Hugging Face Spaces run
# the container as a non-root user that cannot write to the default cache location)
ENV HF_HOME=/tmp/huggingface

EXPOSE 8000

# 0.0.0.0 = listen on all interfaces, so the port is reachable from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
