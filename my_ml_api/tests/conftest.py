import sys
import os
import pytest

# Import modules after modifying sys.path (fixes E402)
from app.model import load_model
from app.config import LOG_LEVEL, MODEL_PATH

# Ensure the app directory is in the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


@pytest.fixture
def model():
    return load_model()


@pytest.fixture
def config():
    return {
        "LOG_LEVEL": LOG_LEVEL,
        "MODEL_PATH": MODEL_PATH,
    }
