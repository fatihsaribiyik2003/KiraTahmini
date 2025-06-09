from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Modelleri ve LabelEncoder nesnelerini yükleme
rf_gezme = joblib.load('rf_gezme_model.pkl')
rf_para = joblib.load('rf_para_model.pkl')
rf_maddi = joblib.load('rf_maddi_model.pkl')
le_gezme = joblib.load('le_gezme.pkl')
le_para = joblib.load('le_para.pkl')
le_maddi = joblib.load('le_maddi.pkl')

def calculate_money_score(gezme_sevgisi, paraya_dikkat, maddi_durum):
    # money_score hesaplama
    score = 0
    
    # Gezme sevgisi katkısı
    if gezme_sevgisi == 'Yüksek':
        score += 0.3
    elif gezme_sevgisi == 'Orta':
        score += 0.15
    
    # Maddi durum katkısı
    if maddi_durum == 'Zengin':
        score += 0.4
    elif maddi_durum == 'Orta':
        score += 0.2
    
    # Paraya dikkat katkısı
    if paraya_dikkat == 'Tutumlu':
        score += 0.1
    elif paraya_dikkat == 'Orta':
        score += 0.05
    
    # Minimum 0.2, maksimum 1.0 sınırı
    score = max(0.2, min(score, 1.0))
    return score

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gelen JSON verisini al
        data = request.get_json()
        
        # 10 soruya verilen cevapları al (1-5 arası)
        answers = [
            data['soru1'], data['soru2'], data['soru3'], data['soru4'], data['soru5'],
            data['soru6'], data['soru7'], data['soru8'], data['soru9'], data['soru10']
        ]
        
        # Veriyi numpy array'e çevir
        answers = np.array([answers])
        
        # Tahmin yap
        gezme_tahmin = le_gezme.inverse_transform(rf_gezme.predict(answers))[0]
        para_tahmin = le_para.inverse_transform(rf_para.predict(answers))[0]
        maddi_tahmin = le_maddi.inverse_transform(rf_maddi.predict(answers))[0]
        
        # money_score hesapla
        money_score = calculate_money_score(gezme_tahmin, para_tahmin, maddi_tahmin)
        
        # Yanıtı JSON olarak döndür
        response = {
            'gezme_sevgisi': gezme_tahmin,
            'paraya_dikkat': para_tahmin,
            'maddi_durum': maddi_tahmin,
            'money_score': round(money_score, 2)
        }
        return jsonify(response), 200
    
    except KeyError as e:
        return jsonify({'error': 'Eksik veya yanlış anahtar: ' + str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Bir hata oluştu: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)