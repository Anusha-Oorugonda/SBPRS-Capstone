import requests
data = {
    "reviews_username": "1234"
  }

url = "http://127.0.0.1:5000/predict_api"
response = requests.post(url, json=data)
print("Sentiment: "+ str(response.json()))
