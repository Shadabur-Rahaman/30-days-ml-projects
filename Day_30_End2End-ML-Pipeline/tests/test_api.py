import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict():
    sample = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert "species" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert data["species"] in ["setosa", "versicolor", "virginica"]

def test_model_info():
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "features" in data
