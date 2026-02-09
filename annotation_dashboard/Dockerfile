FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (no PyTorch)
RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    numpy \
    opencv-python-headless \
    Pillow \
    google-cloud-storage

# Copy application code
COPY config.py app.py ./
COPY models/ models/
COPY utils/ utils/
COPY coding_instructions.md ./

EXPOSE 8080

CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
