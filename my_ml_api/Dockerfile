# Use a lightweight Python image
FROM python:3.9-slim

# Set a working directory inside the container
WORKDIR /app

# Copy and install dependencies separately to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm -rf ~/.cache/pip

# Copy the full application code
COPY . .

# Expose port 8000 for clarity
EXPOSE 8000

# Command to start FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
