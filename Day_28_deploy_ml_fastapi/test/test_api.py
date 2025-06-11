import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "feature1": 45.0,
    "feature2": 2.3,
    "feature3": 0.5
}

response = requests.post(url, json=data)
print("Prediction:", response.json())
