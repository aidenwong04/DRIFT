FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN mkdir -p models && \
    curl -L -o models/drift_dinov2.onnx "https://github.com/aidenwong04/DRIFT/releases/download/v1.0/drift_dinov2.onnx" && \
    curl -L -o models/drift_dinov2.onnx.data "https://github.com/aidenwong04/DRIFT/releases/download/v1.0/drift_dinov2.onnx.data" && \
    curl -L -o models/linear_probe.pth "https://github.com/aidenwong04/DRIFT/releases/download/v1.0/linear_probe.pth"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

