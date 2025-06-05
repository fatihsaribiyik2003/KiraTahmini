import requests

# API adresi
url = 'http://localhost:5000/predict'

# Ev özellikleri
data = {
    'living_room': 3,
    'area': 75,
    'age': 10,
    'floor': 2
}

# POST isteği gönder
try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Hatalı durumlar için hata fırlat
    result = response.json()
    print(f"Tahmini Fiyat: {result['predicted_price']}")
except requests.exceptions.RequestException as e:
    print(f"Hata: {e}")