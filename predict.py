import requests

url = 'http://localhost:9696/predict'

customer = {"job": "student", "duration": 280, "poutcome": "failure"}

response = requests.post(url, json=customer).json()

print(response)