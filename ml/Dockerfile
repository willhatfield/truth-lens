FROM python:3.11-slim
WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt
COPY *.py ./
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8001}
