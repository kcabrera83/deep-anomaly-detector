import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app, _load_models


@pytest.fixture(scope="session", autouse=True)
def ensure_models_loaded():
    _load_models()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
