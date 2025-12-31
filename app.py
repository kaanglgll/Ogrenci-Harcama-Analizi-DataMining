from flask import Flask, render_template, request
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg') # GUI hatası almamak için Agg backend
import matplotlib.pyplot as plt
import io
import base64
import os
import numpy as np

app = Flask(__name__)

# Upload klasör ayarı
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Anket seçenekleri (Sabit Liste)
KNOWN_ITEMS = [
    "Kahve ve Sosyalleşme", "Sigara ve Tütün", "Alkol ve Gece Hayatı",
    "Dijital Oyun", "Dijital Abonelik", "Giyim ve Aksesuar",
    "Kozmetik ve Kişisel Bakım", "Kız/Erkek Arkadaş",
    "Etkinlik/Kültür", "Şans Oyunları",
]

@app.route('/', methods=['GET', 'POST'])
def index():
    rules_data = []
    unique_antecedents = set()
    tree_image = None
    kmeans_image = None
    cluster_summaries = None
    error = None
    active_tab = 'apriori' 

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="Dosya yüklenmedi!")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="Dosya seçilmedi!")

        if file:
            try:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                df = pd.read_excel(filepath, engine='openpyxl')
                action = request.form.get('action')

                # --- ORTAK VERİ HAZIRLIĞI (ID3 ve K-Means için) ---
                if action in ['decision_tree', 'kmeans']:
                    df_encoded = df.copy()
                    # Sütun seçimi (Excel yapısına göre 1'den 6'ya kadar)
                    df_encoded = df_encoded.iloc[:, 1:6]
                    df_encoded.columns = ['Cinsiyet', 'Barınma', 'Ulasim', 'Harcamalar', 'AySonu']
                    df_encoded = df_encoded.dropna()

                    le = LabelEncoder()
                    # Kategorik verileri sayıya çevir
                    df_encoded['Cinsiyet_Code'] = le.fit_transform(df_encoded['Cinsiyet'].astype(str))
                    df_encoded['Barınma_Code'] = le.fit_transform(df_encoded['Barınma'].astype(str))
                    df_encoded['Ulasim_Code'] = le.fit_transform(df_encoded['Ulasim'].astype(str))
                    df_encoded['AySonu_Code'] = le.fit_transform(df_encoded['AySonu'].astype(str))

                    # Harcamaları 1/0'a çevir (One-Hot Encoding Mantığı)
                    for item in KNOWN_ITEMS:
                        df_encoded[item] = df_encoded['Harcamalar'].apply(lambda x: 1 if item in str(x) else 0)
                    
                    # Model Girdisi (X)
                    feature_cols = ['Cinsiyet_Code', 'Barınma_Code', 'Ulasim_Code', 'AySonu_Code'] + KNOWN_ITEMS
                    X = df_encoded[feature_cols]

                # --- MODÜL 1: APRIORI ALGORİTMASI ---
                if action == 'apriori':
                    active_tab = 'apriori'
                    target_col_idx = 4 # Harcamalar sütunu indeksi
                    transactions = []
                    
                    # Veriyi Temizle
                    df_clean = df.dropna(subset=[df.columns[target_col_idx]])
                    
                    # Sepetleri (Transactions) Oluştur
                    for index, row in df_clean.iterrows():
                        current_transaction = []
                        item_text_str = str(row.iloc[target_col_idx])
                        
                        # Bilinen harcamaları ekle
                        for known_item in KNOWN_ITEMS:
                            if known_item in item_text_str: current_transaction.append(known_item)
                        
                        # Virgülle ayrılmış diğerlerini ekle
                        if not current_transaction and len(item_text_str) > 3: 
                            current_transaction = [x.strip() for x in item_text_str.split(',')]
                        
                        # Demografik bilgileri sepete dahil et (Kullanıcı seçtiyse)
                        if request.form.get('use_gender'): current_transaction.append(f"Cinsiyet:{str(row.iloc[1]).strip()}")
                        if request.form.get('use_housing'): current_transaction.append(f"Barınma:{str(row.iloc[2]).strip()}")
                        if request.form.get('use_transport'): current_transaction.append(f"Ulaşım:{str(row.iloc[3]).strip()}")
                        if request.form.get('use_budget'): current_transaction.append(f"AySonu:{str(row.iloc[5]).strip()}")
                        
                        if current_transaction: transactions.append(current_transaction)
                    
                    # Apriori İşlemleri
                    te = TransactionEncoder()
                    te_ary = te.fit(transactions).transform(transactions)
                    df_transformed = pd.DataFrame(te_ary, columns=te.columns_)
                    
                    min_supp = float(request.form.get('min_support', 0.05))
                    min_conf = float(request.form.get('min_confidence', 0.2))
                    
                    frequent_itemsets = apriori(df_transformed, min_support=min_supp, use_colnames=True)
                    
                    if not frequent_itemsets.empty:
                        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
                        for i, row_rule in rules.iterrows():
                            ants = list(row_rule['antecedents'])
                            for item in ants: unique_antecedents.add(item)
                            rules_data.append({
                                'antecedents': ants, 
                                'consequents': list(row_rule['consequents']), 
                                'support': round(row_rule['support'], 2), 
                                'confidence': round(row_rule['confidence'], 2), 
                                'lift': round(row_rule['lift'], 2)
                            })
                        # Varsayılan sıralama: Lift'e göre azalan
                        rules_data.sort(key=lambda x: x['lift'], reverse=True)
                    else: 
                        error = "Belirtilen kriterlere uygun kural bulunamadı. Destek değerini düşürmeyi deneyin."

                # --- MODÜL 2: ID3/CART KARAR AĞACI ---
                elif action == 'decision_tree':
                    active_tab = 'tree'
                    chosen_criterion = request.form.get('tree_criterion', 'entropy')
                    
                    # Konsol Logu
                    print(f"Kullanıcı Tercihi: {chosen_criterion.upper()}")
                    
                    # Hedef ve Girdi Ayrımı
                    y = df_encoded['AySonu_Code']
                    X_tree = X.drop(columns=['AySonu_Code'])

                    # Model Eğitimi
                    clf = DecisionTreeClassifier(criterion=chosen_criterion, max_depth=4, min_samples_leaf=3)
                    clf.fit(X_tree, y)

                    # Görselleştirme
                    plt.figure(figsize=(20, 10))
                    plot_tree(clf, feature_names=list(X_tree.columns), filled=True, rounded=True, fontsize=10)
                    
                    img = io.BytesIO()
                    plt.savefig(img, format='png', bbox_inches='tight')
                    img.seek(0)
                    tree_image = base64.b64encode(img.getvalue()).decode()
                    plt.close()

                # --- MODÜL 3: K-MEANS KÜMELEME ---
                elif action == 'kmeans':
                    active_tab = 'kmeans'
                    n_clusters = int(request.form.get('n_clusters', 3))
                    
                    # Model Eğitimi
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(X) 
                    
                    # Profil Analizi (Cluster Profiling)
                    df_analysis = X.copy()
                    df_analysis['Cluster'] = clusters
                    cluster_summaries = []
                    
                    for i in range(n_clusters):
                        group_data = df_analysis[df_analysis['Cluster'] == i]
                        summary = {
                            'id': i + 1,
                            'count': len(group_data),
                            'features': [],
                            'negatives': []
                        }
                        
                        # Özellik Analizi
                        for item in KNOWN_ITEMS:
                            mean_val = group_data[item].mean()
                            if mean_val >= 0.75: summary['features'].append(f"{item} (%{int(mean_val*100)})")
                            elif mean_val <= 0.10: summary['negatives'].append(item)
                        
                        # Cinsiyet Analizi
                        gender_mean = group_data['Cinsiyet_Code'].mean()
                        if gender_mean < 0.3: summary['features'].append("Çoğunluk: Erkek")
                        elif gender_mean > 0.7: summary['features'].append("Çoğunluk: Kadın")
                        
                        cluster_summaries.append(summary)

                    # Görselleştirme (PCA ile 2 Boyut)
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X)
                    
                    plt.figure(figsize=(10, 6))
                    colors = ['#ff7675', '#74b9ff', '#55efc4', '#a29bfe', '#fdcb6e']
                    
                    for i in range(n_clusters):
                        plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1], 
                                    s=100, c=colors[i % len(colors)], label=f'Grup {i+1}', alpha=0.8, edgecolors='gray')

                    plt.title(f'Öğrenci Profilleri Haritası ({n_clusters} Grup)', fontsize=15)
                    plt.xlabel('Yaşam Tarzı Ekseni (PCA-1)', fontsize=10)
                    plt.ylabel('Harcama Ekseni (PCA-2)', fontsize=10)
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.5)
                    
                    img = io.BytesIO()
                    plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
                    img.seek(0)
                    kmeans_image = base64.b64encode(img.getvalue()).decode()
                    plt.close()

            except Exception as e:
                error = f"Hata oluştu: {str(e)}"

    return render_template('index.html', 
                           rules=rules_data, 
                           error=error, 
                           unique_items=sorted(list(unique_antecedents)),
                           tree_image=tree_image,
                           kmeans_image=kmeans_image,
                           cluster_summaries=cluster_summaries,
                           active_tab=active_tab)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)