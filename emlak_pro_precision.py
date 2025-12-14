import pandas as pd
import streamlit as st
import os
import re
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# --- WEB ARAMA MODÜLÜ (AI ASİSTAN İÇİN) ---
WEB_SEARCH_AKTIF = False
try:
    from duckduckgo_search import DDGS
    WEB_SEARCH_AKTIF = True
except ImportError:
    WEB_SEARCH_AKTIF = False

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="EMLAK PRO MAX: PRECISION", 
    layout="wide", 
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# --- MODERN CSS TASARIM ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    h1, h2, h3 { color: #00e676 !important; font-weight: 800 !important; font-family: 'Segoe UI', sans-serif; }
    
    /* Metrik Kutuları */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        transition: transform 0.3s;
    }
    .metric-container:hover { transform: scale(1.02); border-color: #00e676; }
    .metric-value { font-size: 28px; font-weight: bold; color: #fff; }
    .metric-label { font-size: 12px; color: #00e676; text-transform: uppercase; letter-spacing: 1px; }
    
    /* Tablolar */
    [data-testid="stDataFrame"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; }
    
    /* Yan Menü */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# --- İSTANBUL KOORDİNATLARI (HARİTA İÇİN) ---
ISTANBUL_COORDS = {
    "Adalar": [40.8765, 29.1325], "Arnavutköy": [41.1856, 28.7402], "Ataşehir": [40.9932, 29.1132],
    "Avcılar": [40.9789, 28.7231], "Bağcılar": [41.0343, 28.8576], "Bahçelievler": [41.0001, 28.8601],
    "Bakırköy": [40.9832, 28.8732], "Başakşehir": [41.0976, 28.8071], "Bayrampaşa": [41.0354, 28.9123],
    "Beşiktaş": [41.0428, 29.0076], "Beykoz": [41.1213, 29.0963], "Beylikdüzü": [40.9892, 28.6434],
    "Beyoğlu": [41.0284, 28.9736], "Büyükçekmece": [41.0321, 28.5872], "Çatalca": [41.1432, 28.4593],
    "Çekmeköy": [41.0351, 29.1751], "Esenler": [41.0487, 28.8856], "Esenyurt": [41.0342, 28.6801],
    "Eyüpsultan": [41.0471, 28.9332], "Fatih": [41.0102, 28.9403], "Gaziosmanpaşa": [41.0581, 28.9124],
    "Güngören": [41.0253, 28.8651], "Kadıköy": [40.9901, 29.0254], "Kağıthane": [41.0812, 28.9753],
    "Kartal": [40.8901, 29.1901], "Küçükçekmece": [41.0002, 28.7801], "Maltepe": [40.9241, 29.1311],
    "Pendik": [40.8801, 29.2501], "Sancaktepe": [40.9905, 29.2201], "Sarıyer": [41.1681, 29.0572],
    "Silivri": [41.0742, 28.2471], "Sultanbeyli": [40.9654, 29.2673], "Sultangazi": [41.1071, 28.8681],
    "Şile": [41.1754, 29.6101], "Şişli": [41.0601, 28.9876], "Tuzla": [40.8401, 29.3201],
    "Ümraniye": [41.0256, 29.0963], "Üsküdar": [41.0261, 29.0152], "Zeytinburnu": [40.9904, 28.9001]
}

# --- VERİ YÜKLEME VE TEMİZLEME ---
@st.cache_data
def load_data(file_path):
    try:
        if isinstance(file_path, str):
            try: df = pd.read_csv(file_path, sep=None, engine='python')
            except: df = pd.read_csv(file_path, sep=',')
        else:
            df = pd.read_csv(file_path, sep=None, engine='python')

        df.columns = df.columns.str.strip()
        
        # Sütun Eşleştirme (Regex ile Otomatik Bulma)
        def find_col(keywords):
            for col in df.columns:
                if any(k.lower() in col.lower() for k in keywords): return col
            return None

        col_dist = find_col(['district', 'ilçe'])
        col_neigh = find_col(['neighborhood', 'mahalle', 'semt'])
        col_price = find_col(['price', 'fiyat'])
        col_m2 = find_col(['m² (net)', 'net m2', 'net']) or find_col(['m² (brüt)', 'gross'])
        col_room = find_col(['room', 'oda']) # Oda sayısı için
        
        if not (col_dist and col_price and col_m2): return None, "Kritik sütunlar (İlçe, Fiyat, m2) bulunamadı.", None

        # Sütunları İngilizce (Kod İçi) İsimlere Çevir
        rename_map = {col_dist: 'District', col_price: 'Price', col_m2: 'm²'}
        if col_neigh: rename_map[col_neigh] = 'Neighborhood'
        if col_room: rename_map[col_room] = 'Oda_Text'
        
        df = df.rename(columns=rename_map)

        # Sayısal Temizlik
        def clean_num(val):
            if isinstance(val, str): 
                val = re.sub(r'[^\d.]', '', val.replace('TL', '').replace('.', '').replace(',', '.'))
            try: return float(val)
            except: return np.nan

        df['Price'] = df['Price'].apply(clean_num)
        df['m²'] = df['m²'].apply(clean_num)
        
        # 1. Temel Temizlik (Boşlar ve Sıfırlar)
        df = df.dropna(subset=['Price', 'm²'])
        df = df[(df['Price'] > 5000) & (df['m²'] > 10)]
        
        # 2. İleri Düzey Temizlik (Aykırı Değerler - Outliers)
        # Çok aşırı ucuz veya çok aşırı pahalı (hatalı girilmiş) verileri modelden uzak tutuyoruz.
        # Bu işlem R2 Skorunu %80 üzerine çıkarmak için kritiktir.
        df['Birim_Fiyat'] = df['Price'] / df['m²']
        
        # Alt %1 ve Üst %1'lik dilimi (Çöp Veri) at
        lower_bound = df['Birim_Fiyat'].quantile(0.01)
        upper_bound = df['Birim_Fiyat'].quantile(0.99)
        df = df[(df['Birim_Fiyat'] >= lower_bound) & (df['Birim_Fiyat'] <= upper_bound)]

        # İlçe İsim Düzeltme
        df['District'] = df['District'].str.title().str.strip()
        
        # Oda Sayısı Parse Etme (3+1 -> 3)
        def parse_room(val):
            try:
                nums = re.findall(r'\d+', str(val))
                return int(nums[0]) if nums else 2
            except: return 2
            
        if 'Oda_Text' in df.columns:
            df['Oda_Sayisi'] = df['Oda_Text'].apply(parse_room)
        else:
            df['Oda_Sayisi'] = 2 # Varsayılan

        # Özellik Sütunlarını Yakala (Havuz, Metro vs.)
        feature_cols = []
        keywords = ['havuz', 'pool', 'metro', 'güvenlik', 'security', 'otopark', 'garage', 'balkon', 'asansör', 'deniz', 'boğaz', 'teras']
        for col in df.columns:
            if any(k in col.lower() for k in keywords) and col not in rename_map.values():
                # 1/0 Dönüşümü
                df[col] = df[col].fillna(0).apply(lambda x: 1 if str(x).lower() in ['1', 'yes', 'var', 'true', 'evet'] else 0)
                feature_cols.append(col)

        return df, None, feature_cols

    except Exception as e: return None, str(e), None

# --- MAKİNE ÖĞRENMESİ (EĞİTİM & TAHMİN) ---
def train_model_and_predict(df, feature_cols):
    
    # 1. Target Encoding (En Önemli Adım)
    # Modelin "Semt" değerini anlaması için, her semte ortalama m2 fiyatı kadar puan veriyoruz.
    if 'Neighborhood' in df.columns:
        neigh_map = df.groupby('Neighborhood').apply(lambda x: (x['Price']/x['m²']).median()).to_dict()
        df['Konum_Skoru'] = df['Neighborhood'].map(neigh_map)
        # Bilinmeyen semtler için ilçe ortalamasını kullan
        dist_map = df.groupby('District').apply(lambda x: (x['Price']/x['m²']).median()).to_dict()
        df['Konum_Skoru'] = df['Konum_Skoru'].fillna(df['District'].map(dist_map))
    else:
        dist_map = df.groupby('District').apply(lambda x: (x['Price']/x['m²']).median()).to_dict()
        df['Konum_Skoru'] = df['District'].map(dist_map)

    # 2. Model Verisi Hazırla
    # Girdiler: m2, Oda Sayısı, Konum Skoru ve Ekstra Özellikler
    features = ['m²', 'Oda_Sayisi', 'Konum_Skoru'] + feature_cols
    X = df[features].fillna(0)
    
    # Logaritmik Dönüşüm (Fiyatlar arasındaki uçurumu dengeler, doğruluğu artırır)
    y = np.log1p(df['Price'])

    # 3. Eğitim (Gradient Boosting - Yüksek Hassasiyet)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 4. Performans Ölçümü
    y_pred = model.predict(X_test)
    r2 = r2_score(np.expm1(y_test), np.expm1(y_pred)) # Gerçek değerlerle skor

    # 5. Tüm Veriye Uygula (Tahminleri Üret)
    df['Tahmini_Fiyat'] = np.expm1(model.predict(X))
    df['Sapma_%'] = ((df['Price'] - df['Tahmini_Fiyat']) / df['Tahmini_Fiyat']) * 100
    
    # Durum Belirleme (Hata Ayıklama)
    # Modelin %90 doğru olduğunu varsayarsak, %30'dan fazla sapmalar "Anormal"dir.
    conditions = [
        (df['Sapma_%'] < -35), # Fiyat Tahminden %35 daha ucuz -> HATA (veya Çok Kötü Durumda)
        (df['Sapma_%'].between(-35, -10)), # Fiyat Tahminden %10-%35 ucuz -> FIRSAT
        (df['Sapma_%'].between(-10, 20)), # Normal Piyasa
        (df['Sapma_%'] > 20) # Fiyat Tahminden %20 pahalı -> PAHALI/HATALI
    ]
    choices = ['⛔ HATALI (Çok Düşük)', '💎 SÜPER FIRSAT', '✅ PİYASA UYGUN', '❌ PAHALI / HATALI']
    df['Durum'] = np.select(conditions, choices, default='✅ PİYASA UYGUN')

    return df, r2

# --- ARAYÜZ (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1040/1040993.png", width=80)
    st.title("EMLAK PRO MAX")
    st.caption("Yüksek Doğruluklu Değerleme Sistemi")
    
    uploaded_file = st.file_uploader("Veri Seti (CSV)", type=['csv'])
    if not uploaded_file and os.path.exists("veri.csv"): uploaded_file = "veri.csv"

    if uploaded_file:
        df_raw, err, feats = load_data(uploaded_file)
        if err: st.error(err); st.stop()
        
        st.success(f"✅ {len(df_raw)} Temiz İlan Hazır")
        
        if st.button("🚀 MODELİ EĞİT VE ANALİZ ET", type="primary"):
            with st.spinner("Gradient Boosting Modeli Eğitiliyor (Bu işlem işlemciye bağlı birkaç saniye sürebilir)..."):
                df_res, score = train_model_and_predict(df_raw, feats)
                st.session_state['data'] = df_res
                st.session_state['score'] = score
                st.session_state['feats'] = feats
                st.rerun()
    else:
        st.info("Başlamak için CSV yükleyin.")

# --- ANA EKRAN ---
if 'data' in st.session_state:
    df = st.session_state['data']
    r2 = st.session_state['score']
    
    # METRİKLER
    c1, c2, c3, c4 = st.columns(4)
    opp_count = len(df[df['Durum'] == '💎 SÜPER FIRSAT'])
    err_count = len(df[df['Durum'].str.contains('HATALI')])
    
    # Skor Rengi
    score_color = "#00e676" if r2 > 0.80 else "#ffab00"
    
    with c1: st.markdown(f'<div class="metric-container" style="border-color:{score_color}"><div class="metric-value" style="color:{score_color}">%{r2*100:.1f}</div><div class="metric-label">Model Doğruluğu (R²)</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-container"><div class="metric-value">{len(df)}</div><div class="metric-label">Toplam İlan</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-container" style="border-color:#00e676"><div class="metric-value" style="color:#00e676">{opp_count}</div><div class="metric-label">Fırsat Sayısı</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-container" style="border-color:#ff1744"><div class="metric-value" style="color:#ff1744">{err_count}</div><div class="metric-label">Aykırı / Hatalı Veri</div></div>', unsafe_allow_html=True)

    st.write("")
    
    # SEKMELER
    tabs = st.tabs(["🗺️ İSTANBUL HARİTASI", "📊 DETAYLI ANALİZ", "💎 FIRSATLAR & HATALAR", "🤖 AI ASİSTAN"])

    # 1. HARİTA (EN UYGUN YERLER)
    with tabs[0]:
        st.markdown("### 🗺️ İstanbul Bölgesel Fırsat Haritası")
        st.caption("Yeşil: Fırsat Bölgesi (Modelden Ucuz) | Kırmızı: Pahalı Bölgesi | Boyut: İlan Yoğunluğu")
        
        # İlçe Bazlı Özet
        dist_summary = df.groupby('District').agg({
            'Price': 'mean', 
            'm²': 'count', 
            'Sapma_%': 'mean' # Negatif sapma = Fırsat bölgesi
        }).reset_index()
        
        try:
            m = folium.Map(location=[41.0082, 28.9784], zoom_start=10, tiles="CartoDB dark_matter")
            
            for idx, row in dist_summary.iterrows():
                dist_name = row['District']
                # Koordinat Eşleştirme
                coords = None
                for key, val in ISTANBUL_COORDS.items():
                    if key.lower() in dist_name.lower():
                        coords = val
                        break
                
                if coords:
                    # Renk Skalası (Fırsat Durumuna Göre)
                    if row['Sapma_%'] < -5: color = "#00e676" # Yeşil (Fırsat Bölgesi)
                    elif row['Sapma_%'] > 5: color = "#ff1744" # Kırmızı (Pahalı)
                    else: color = "#29b6f6" # Mavi (Normal)
                    
                    folium.CircleMarker(
                        location=coords,
                        radius=5 + (row['m²'] / dist_summary['m²'].max() * 20),
                        popup=f"<b>{dist_name}</b><br>Ort. Fiyat: {row['Price']:,.0f} TL<br>Fırsat Skoru: {row['Sapma_%']:.1f}%",
                        color=color, fill=True, fill_color=color, fill_opacity=0.6
                    ).add_to(m)
            
            st_folium(m, width="100%", height=500)
        except Exception as e: st.error(f"Harita hatası: {e}")

    # 2. DETAYLI ANALİZ
    with tabs[1]:
        st.markdown("### 🔍 Model Tahminleri vs Gerçek Fiyatlar")
        # KeyError ÇÖZÜMÜ: Sütun adını 'Price' olarak kullanıyoruz, gösterirken 'Fiyat' yazıyoruz.
        st.dataframe(
            df[['District', 'm²', 'Price', 'Tahmini_Fiyat', 'Sapma_%', 'Durum']],
            column_config={
                "Price": st.column_config.NumberColumn("İlan Fiyatı", format="%d TL"),
                "Tahmini_Fiyat": st.column_config.NumberColumn("Olması Gereken (AI)", format="%d TL"),
                "Sapma_%": st.column_config.NumberColumn("Fark", format="%.1f%%"),
            },
            use_container_width=True, height=600
        )

    # 3. FIRSATLAR VE HATALAR
    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.success("💎 **YATIRIMLIK FIRSATLAR**")
            st.caption("Yapay zekanın 'Bu özelliklere göre bu ev ucuz kalmış' dedikleri.")
            opportunities = df[df['Durum'] == '💎 SÜPER FIRSAT'].sort_values(by='Sapma_%')
            st.dataframe(opportunities[['District', 'm²', 'Price', 'Tahmini_Fiyat', 'Sapma_%']], use_container_width=True)
            
        with c2:
            st.error("🚫 **HATALI / AYKIRI VERİLER**")
            st.caption("Piyasa gerçeklerinden çok uzak (Muhtemelen veri giriş hatası) ilanlar.")
            errors = df[df['Durum'].str.contains('HATALI')]
            st.dataframe(errors[['District', 'm²', 'Price', 'Tahmini_Fiyat', 'Durum']], use_container_width=True)

    # 4. AI ASİSTAN
    with tabs[3]:
        st.markdown("### 🤖 Emlak Zekası")
        chat_col, info_col = st.columns([3, 1])
        
        with info_col:
            st.info("Bu asistan, internet üzerinden güncel kredi oranlarını, bölge trendlerini ve tapu süreçlerini araştırabilir.")
        
        with chat_col:
            chat_cont = st.container(height=400, border=True)
            if "messages" not in st.session_state: st.session_state.messages = []
            
            for msg in st.session_state.messages:
                chat_cont.chat_message(msg["role"]).write(msg["content"])
                
            if prompt := st.chat_input("Soru sor (Örn: Şu an konut kredisi faizleri kaç?)"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                chat_cont.chat_message("user").write(prompt)
                
                response = "İnternet modülü kapalı."
                if WEB_SEARCH_AKTIF:
                    with chat_cont.chat_message("assistant"):
                        with st.spinner("Araştırıyorum..."):
                            try:
                                with DDGS() as ddgs:
                                    results = list(ddgs.text(prompt, region='tr-tr', max_results=2))
                                    if results:
                                        response = "**Bulduğum Bilgiler:**\n\n" + "\n\n".join([f"- {r['body']} ([Kaynak]({r['href']}))" for r in results])
                                    else:
                                        response = "Üzgünüm, güncel bir bilgi bulamadım."
                            except: response = "Bağlantı hatası oluştu."
                        st.write(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("👈 Analize başlamak için sol menüden veri yükleyin ve butona basın.")
