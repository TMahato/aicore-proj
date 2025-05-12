# Specify which base layers (default dependencies) to use
# Using a specific Python version for stability
FROM python:3.12-slim

# Creates directory within your Docker image
RUN mkdir -p /app/src/
# Don't place anything in below folders yet, just create them
RUN mkdir -p /app/data/
RUN mkdir -p /app/model/

# Install system dependencies required by some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copies file from your Local system TO path in Docker image
COPY requirements.txt /app/src/ 
COPY main.py /app/src/
 
# Installs dependencies within your Docker image
RUN pip3 install --no-cache-dir -r /app/src/requirements.txt

# Enable permission to execute anything inside the folder app
RUN chgrp -R 65534 /app && \
    chmod -R 777 /app

# Set working directory
WORKDIR /app/src

# Command to run when container starts
CMD ["python", "main.py"]