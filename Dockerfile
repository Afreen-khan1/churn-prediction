# Use official Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for scikit-learn, pandas, etc.
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for uploads and graphs (important!)
RUN mkdir -p uploads static/graphs && chmod -R 755 uploads static/graphs

# Hugging Face Spaces uses port 7860
ENV PORT=7860

# Expose the port
EXPOSE $PORT

# Run the app with gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120 app:app
