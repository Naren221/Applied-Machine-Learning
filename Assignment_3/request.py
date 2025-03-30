import requests

url = "http://127.0.0.1:5000/score"
data = {"text": "k i will sent it again"}

response = requests.post(url, json=data)
print(response.json())  # Output: {"number": 5, "square": 25}
