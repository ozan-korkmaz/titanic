# titanic
# Titanic EDA Dashboard 🚢📊

Bu proje, [Kaggle Titanic veri seti](https://www.kaggle.com/c/titanic) üzerinde yapılan **Keşifsel Veri Analizi (EDA)** sonuçlarını görselleştiren basit bir **Flask tabanlı web uygulaması**dır.  

Grafikler **matplotlib** ve **seaborn** ile üretilir ve Flask ile yerel bir web dashboard üzerinde sunulur.  

---

## 🚀 Özellikler
- `train.csv` ve `test.csv` dosyalarını yükleyerek otomatik analiz.
- Cinsiyet, sınıf, yaş, aile büyüklüğü gibi faktörlere göre **hayatta kalma oranları**.
- **Korelasyon matrisi** (heatmap).
- **Feature Importance** (RandomForest Classifier ile basit önem dereceleri).
- Responsive ve modern HTML/CSS arayüzü.

---

---

## ⚙️ Kurulum

1. Gerekli kütüphaneleri yükle:
   ```bash
   pip install flask pandas matplotlib seaborn scikit-learn
2.Proje klasöründe train.csv, test.csv dosyalarının bulunduğundan emin ol.

3. Flask uygulamasını başlat:
`python flask_titanic_dashboard.py
`
4. Tarayıcıda aç:
`http://127.0.0.1:5000
`
📊 Dashboard Ekran Görüntüleri
+Cinsiyete Göre Hayatta Kalma
+Yolcu Sınıfına Göre Hayatta Kalma
+Yaş Dağılımı ve KDE
+Aile Büyüklüğüne Göre Hayatta Kalma
+Korelasyon Matrisi
+Feature Importance (RandomForest)

🛠 Kullanılan Teknolojiler
Python 3
Flask → Web framework
Pandas → Veri analizi
Matplotlib & Seaborn → Görselleştirme
Scikit-learn → Makine öğrenmesi (RandomForest)


📌 Notlar
Bu uygulama öğrenme/portföy amaçlı geliştirilmiştir.
Geliştirme ortamı için uygundur, prodüksiyon için WSGI (Gunicorn, Nginx) önerilir.
Daha interaktif grafikler için Plotly / Altair entegre edilebilir.
