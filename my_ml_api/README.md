# Fraud Detection ML API

## Overview
This project implements a FastAPI-based machine learning API that:
1. Accepts individual JSON requests with fields: `timestamp`, `session_id`, `device`, and `price`.
2. Maintains a session cache for incoming requests.
3. Uses `FraudDetectionDataset` for feature engineering.
4. Generates predictions using a pre-trained ML model.

## Setup Instructions

### 1. Create a Conda Environment

```
conda create --name fraud_detection_env python=3.9 -y
conda activate fraud_detection_env
```

### 2. Install Dependencies

Install all required dependencies using `pip`:
```
pip install -r requirements.txt
```

Alternatively, install dependencies manually:
```
pip install fastapi==0.115.7 uvicorn==0.34.0 pandas==2.2.3 numpy==2.0.2 pytest==8.3.4 pytest-cov==6.0.0 scikit-learn==1.6.1 xgboost==2.1.3 pydantic==2.10.6 mypy==1.14.1 black==25.1.0 flake8==7.1.1 pre-commit==4.1.0
```

### 3. Run the ML API Locally

**Run the API**
Start the FastAPI application using `Uvicorn`:
```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```


**Example: Single Request (No Cache History)**
Using `curl`:
```
curl -X POST "http://127.0.0.1:8000/v1/predict" -H "Content-Type: application/json" -d '{
    "timestamp": "2025-01-01T12:00:00",
    "session_id": "session_1",
    "device": "mobile",
    "price": 100.0
}'
```

Expected Response:
```
{
    "prediction": 0.15885697305202484
}
```

#### Example: Session-Based Using Redis to requests with Cache 
##### (Not Implemented)

How will it work in real-life to store features of previous requests from the same
session.

This example demonstrates how Redis stores previous session transactions to enhance feature generation.

**3.a. Install and Start Redis (Linux/WSL)**
```
sudo apt update
sudo apt install redis
redis-server
```
Ensure Redis is running:
```
redis-cli ping
```
**Expected output:** PONG

**3.b. Run FastAPI with Redis Support**
```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**3.c. First Request in a Session (session_1)**
This request initializes session-based tracking in Redis.
```
curl -X POST "http://127.0.0.1:8000/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "timestamp": "2025-01-01T12:00:00",
         "session_id": "session_1",
         "device": "android",
         "price": 100.0
     }'
```

**3.d. Second Request in the Same Session (session_1)**
Now, past transactions are used to compute enhanced session-based features.
```
curl -X POST "http://127.0.0.1:8000/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "timestamp": "2025-01-01T12:05:00",
         "session_id": "session_1",
         "device": "android",
         "price": 150.0
     }'
```

### 4. Code Quality Checks

**Type Checking with mypy**
Run `mypy` to ensure proper type hints:
```
mypy app/
```

**Code Formatting with black**
Ensure consistent code style with `black`:
```
black --line-length 79 app/ tests/
```

**Linting with `flake8`**
Check for code style errors with `flake8`:
```
flake8 app/ tests/
```

### 5. Pre-commit Hooks Setup

Run pre-commit manually on all files:
```
pre-commit run --all-files
```

### 6. Testing

Run unit tests with `pytest`:
```
pytest tests/
```
Generate a test coverage report:
```
pytest --cov=app tests/
```

### 7. Dockerizing the Application

**Build the Docker Image**
To containerize the application, first ensure you have Docker installed, then build the image:
```
docker build -t fraud_detection_api
```
**Run the Docker Container**
Run the container with the following command
```
docker run -p 8000:8000 fraud_detection_api
```

**Test the API Inside the Docker Container**
Once the container is running, use the following command to test the API
```
curl -X POST "http://localhost:8000/v1/predict" -H "Content-Type: application/json" -d '{
    "timestamp": "2025-01-01T12:00:00",
    "session_id": "123",
    "device": "mobile",
    "price": 100.0
}'
```

**Stop and Remove the Docker Container**
To stop the container, run:
```
docker ps  # Find the container ID

docker stop <container_id>
docker rm <container_id>
```

### 8. Deactivate the Conda Environment
When you're done, you can deactivate the environment with:

```
conda deactivate
```

### 9. Remove the Conda Environment
If you want to remove the Conda environment later:

```
conda remove --name fraud_detection_env --all
```

### 10. Code Review Practices

- Use `mypy` for type hints.

- Format code with `black`.

- Enforce style guidelines with `flake8`.

- Use `pre-commit` hooks to automate checks.

- Ensure proper test coverage with `pytest`.

- Perform regular code reviews before merging changes.