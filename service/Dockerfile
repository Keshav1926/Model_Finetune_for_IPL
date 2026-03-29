# Use the Hugging Face GPU base image
FROM huggingface/transformers-pytorch-gpu:latest

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Copy your app files
COPY requirements.txt .
COPY app.py .

# Install extra dependencies (most are preinstalled)
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Default envs (can be overridden at runtime)
ENV MODEL_REPO="keshav1926s/phi3-lora-adapter"
ENV HF_TOKEN="hf_YOUR_HF_TOKEN_HERE"
ENV BASE_MODEL="microsoft/phi-3-mini-4k-instruct"
ENV USE_4BIT="true"
ENV DEVICE="cuda"
ENV PORT=1926

EXPOSE 1926

# Launch FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "1926", "--workers", "1"]

