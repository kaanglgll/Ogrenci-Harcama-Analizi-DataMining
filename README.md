# 🎓 Öğrenci Harcama Analizi ve Veri Madenciliği Platformu

Bu proje, üniversite öğrencilerinin harcama alışkanlıklarını analiz etmek için geliştirilmiş web tabanlı bir veri madenciliği uygulamasıdır. Python (Flask) kullanılarak geliştirilmiştir.

## 🚀 Özellikler

Proje içerisinde 3 farklı veri madenciliği algoritması aktif olarak çalışmaktadır:

1.  **Apriori Algoritması (Birliktelik Analizi):** * Öğrencilerin hangi harcamaları birlikte yaptığını keşfeder (Örn: "Sigara içenler %80 ihtimalle Kahve de içiyor").
    * *Support, Confidence ve Lift* değerlerine göre filtreleme imkanı sunar.

2.  **ID3 & CART Karar Ağaçları (Sınıflandırma):**
    * Öğrencinin "Ay sonunu getirip getiremeyeceğini" tahmin eder.
    * **Entropy** ve **Gini** kriterlerine göre ağaç oluşturabilir.

3.  **K-Means (Kümeleme):**
    * Öğrencileri harcama ve yaşam tarzlarına göre otomatik olarak gruplara (segmentlere) ayırır.
    * Her grubun karakteristik özelliklerini (Örn: "Gece Hayatı Sevenler", "Tasarrufçular") raporlar.

## 🛠️ Kurulum

Projeyi kendi bilgisayarınızda çalıştırmak için:

1.  Repoyu klonlayın:
    ```bash
    git clone [https://github.com/KULLANICI_ADIN/REPO_ADIN.git](https://github.com/KULLANICI_ADIN/REPO_ADIN.git)
    ```
2.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```
3.  Uygulamayı başlatın:
    ```bash
    python app.py
    ```
4.  Tarayıcıda `http://127.0.0.1:5000` adresine gidin.

## 📷 Ekran Görüntüleri
<img width="1863" height="883" alt="image" src="https://github.com/user-attachments/assets/b3e7eaee-be7c-4b8e-8ebd-b2a57784e8d4" />
<img width="1344" height="756" alt="image" src="https://github.com/user-attachments/assets/d1be02f9-28bb-4157-9d58-48895da8570b" />
<img width="827" height="790" alt="image" src="https://github.com/user-attachments/assets/28e7def5-f210-48fd-8666-22878bbc94fe" />


