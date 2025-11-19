FROM python:3.11-slim
WORKDIR /app

# Copy and install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else (this invalidates cache when code changes, but that's fine)
COPY . .

EXPOSE 8001
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8001"]