# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Ensure Python can find your modules
ENV PYTHONPATH=/app

# Install dependencies first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Expose FastAPI port
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]