
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)

# Modeli yükle
try:
    model = tf.keras.models.load_model('ev_fiyat_tahmin_modeli.h5')
except Exception as e:
    print(f"Model yükleme hatası: {e}")
    exit(1)

# Scaler'ları yükle
try:
    with open('scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
except Exception as e:
    print(f"Scaler yükleme hatası: {e}")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # İstekten JSON verisini al
        data = request.get_json()
        
        # Gerekli alanları kontrol et
        required_fields = ['living_room', 'area', 'age', 'floor']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Eksik alanlar var: living_room, area, age, floor gerekli'}), 400
        
        # Girdi verisini hazırla
        input_data = np.array([[
            float(data['living_room']),
            float(data['area']),
            float(data['age']),
            float(data['floor'])
        ]])
        
        # Girdiyi ölçeklendir
        input_scaled = scaler_X.transform(input_data)
        
        # Tahmin yap
        prediction_scaled = model.predict(input_scaled, verbose=0)
        
        # Tahmini orijinal ölçeğe çevir
        prediction = scaler_y.inverse_transform(prediction_scaled)
        
        # Sonucu döndür
        return jsonify({
            'predicted_price': float(prediction[0][0]),
            'status': 'başarılı'
        }), 200
    
    except ValueError as ve:
        return jsonify({'error': f'Geçersiz veri formatı: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Sunucu hatası: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API çalışıyor'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
