import pandas as pd
import streamlit as st
import os

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Emlak Piyasası Analizi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- AYARLAR ---
# Bu dosya ismini, senin 27.000 satırlık dosyanın ismiyle aynı yap veya dosyanın adını bu yap.
VARSAYILAN_DOSYA_ADI = "veri.csv" 

# --- 1. VERİ YÜKLEME VE TEMİZLEME FONKSİYONU ---
@st.cache_data
def load_data(file_path_or_buffer):
    """Hem dosya yolundan (str) hem de yüklenen dosyadan (buffer) veri okur."""
    try:
        # Dosya yolundan mı yoksa yüklenen dosyadan mı okuyoruz?
        if isinstance(file_path_or_buffer, str):
            # Varsayılan dosyayı okurken ayırıcıyı otomatik algılamaya çalışalım
            try:
                df = pd.read_csv(file_path_or_buffer, sep=None, engine='python')
            except:
                df = pd.read_csv(file_path_or_buffer, sep=',') # Virgül dene
        else:
            # Kullanıcı dosya yükledi
            df = pd.read_csv(file_path_or_buffer, sep=None, engine='python')

        # Sütun isimlerindeki boşlukları temizle
        df.columns = df.columns.str.strip()

        # --- AKILLI SÜTUN EŞLEŞTİRME ---
        # Veri setindeki sütun isimleri farklı olabilir, standartlaştıralım.
        cols = df.columns.str.lower()
        
        # İlçe Sütunu Bul
        col_dist = next((c for c in df.columns if 'district' in c.lower() or 'ilçe' in c.lower() or 'semt' in c.lower()), None)
        # Fiyat Sütunu Bul
        col_price = next((c for c in df.columns if 'price' in c.lower() or 'fiyat' in c.lower() or 'bedel' in c.lower()), None)
        # m2 Sütunu Bul
        col_m2 = next((c for c in df.columns if 'net' in c.lower() or 'm2' in c.lower() or 'm²' in c.lower()), None)

        if not (col_dist and col_price and col_m2):
            return None, f"Gerekli sütunlar bulunamadı. Bulunanlar: {list(df.columns)}"

        # Sadece gerekli veriyi al ve yeniden adlandır
        df_clean = df[[col_dist, col_price, col_m2]].copy()
        df_clean.columns = ['District', 'Price', 'm²_Net']

        # --- TEMİZLİK ---
        # Fiyat Temizliği
        def clean_price(val):
            if isinstance(val, str):
                val = val.replace('TL', '').replace('.', '').replace(',', '').strip()
            try:
                return float(val)
            except:
                return 0

        # m2 Temizliği
        def clean_m2_val(val):
            if isinstance(val, str):
                val = val.lower().replace('m2', '').replace('m²', '').strip()
            try:
                return float(val)
            except:
                return 0

        df_clean['Price'] = df_clean['Price'].apply(clean_price)
        df_clean['m²_Net'] = df_clean['m²_Net'].apply(clean_m2_val)

        # Filtreleme (Hatalı verileri at)
        df_clean = df_clean[df_clean['Price'] > 1000]
        df_clean = df_clean[df_clean['m²_Net'] > 10]

        # Birim Fiyat
        df_clean['Birim_Fiyat'] = df_clean['Price'] / df_clean['m²_Net']
        
        return df_clean, None

    except Exception as e:
        return None, str(e)

# --- 2. UYGULAMA MANTIĞI ---

st.title("🏙️ Gayrimenkul Veri Analiz Platformu")
st.markdown("Bu platform, 27.000+ satırlık veri seti üzerinde anlık piyasa analizi yapar.")

# SIDEBAR - VERİ KAYNAĞI
st.sidebar.header("📂 Veri Kaynağı")

uploaded_file = st.sidebar.file_uploader("Kendi CSV dosyanızı yüklemek ister misiniz?", type=['csv'])

df_global = None
error_msg = None

# MANTIK: Dosya yüklendiyse onu kullan, yüklenmediyse klasördeki varsayılanı kullan.
if uploaded_file is not None:
    st.sidebar.info("Kullanıcı dosyası analiz ediliyor...")
    df_global, error_msg = load_data(uploaded_file)
else:
    # Varsayılan dosya kontrolü
    if os.path.exists(VARSAYILAN_DOSYA_ADI):
        st.sidebar.success(f"✅ Hazır veritabanı kullanılıyor: {VARSAYILAN_DOSYA_ADI}")
        df_global, error_msg = load_data(VARSAYILAN_DOSYA_ADI)
    else:
        st.error(f"⚠️ HATA: '{VARSAYILAN_DOSYA_ADI}' dosyası bulunamadı!")
        st.info("Lütfen 27.000 satırlık CSV dosyanızı, bu Python dosyasıyla aynı klasöre koyun ve adını 'veri.csv' yapın.")
        st.stop()

if error_msg:
    st.error(f"Veri işlenirken hata: {error_msg}")
    st.stop()

if df_global is None or df_global.empty:
    st.warning("Veri seti boş veya okunamadı.")
    st.stop()

# --- 3. ANALİZ PANELİ (Veri Başarıyla Yüklendiyse Burası Çalışır) ---

all_districts = sorted(df_global['District'].unique().tolist())

# Filtreler
st.sidebar.header("🔎 Analiz Filtreleri")
main_district = st.sidebar.selectbox("Hedef Bölge Seçin:", all_districts, index=0)

comp_districts = [d for d in all_districts if d != main_district]
# Varsayılan olarak mantıklı 3 bölge seçelim (yoksa ilk 3)
compare_selection = st.sidebar.multiselect(
    "Kıyaslanacak Bölgeler:", 
    comp_districts,
    default=comp_districts[:3] if len(comp_districts) >= 3 else comp_districts
)

# --- İSTATİSTİKLER ---

# Verileri hazırla
selected_districts = [main_district] + compare_selection
df_filtered = df_global[df_global['District'].isin(selected_districts)]

# Gruplama
stats = df_filtered.groupby('District').agg(
    İlan_Sayısı=('Price', 'count'),
    Ort_Fiyat=('Price', 'mean'),
    Ort_m2=('m²_Net', 'mean'),
    Birim_Fiyat=('Birim_Fiyat', 'mean')
).sort_values('Birim_Fiyat')

# Hedef Bölge Metrikleri
target_stats = stats.loc[main_district]

st.divider()
st.header(f"📍 {main_district} Piyasa Özeti")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Toplam İlan", f"{int(target_stats['İlan_Sayısı']):,} Adet")
kpi2.metric("Ortalama Fiyat", f"{target_stats['Ort_Fiyat']:,.0f} TL")
kpi3.metric("Ortalama m²", f"{target_stats['Ort_m2']:.0f} m²")
kpi4.metric("m² Birim Fiyatı", f"{target_stats['Birim_Fiyat']:,.2f} TL")

# --- KARŞILAŞTIRMA VE GRAFİK ---

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Bölgesel Fiyat Karşılaştırması")
    st.bar_chart(stats['Birim_Fiyat'], color="#007bff") 

with col_right:
    st.subheader("📋 Detaylı Tablo")
    # Tabloyu güzelleştir
    display_df = stats.copy()
    display_df['Ort_Fiyat'] = display_df['Ort_Fiyat'].apply(lambda x: f"{x:,.0f} TL")
    display_df['Birim_Fiyat'] = display_df['Birim_Fiyat'].apply(lambda x: f"{x:,.2f} TL")
    display_df['Ort_m2'] = display_df['Ort_m2'].apply(lambda x: f"{x:.0f} m²")
    st.dataframe(display_df[['Ort_Fiyat', 'Birim_Fiyat', 'İlan_Sayısı']], use_container_width=True)

# --- YAPAY ZEKA ÖNERİSİ ---
st.divider()
st.subheader("🧠 Yapay Zeka Tavsiyesi")

cheapest_district = stats.index[0]
cheapest_price = stats.iloc[0]['Birim_Fiyat']
target_price = target_stats['Birim_Fiyat']

if cheapest_district == main_district:
    st.success(f"✅ **Alım Fırsatı:** Seçtiğiniz **{main_district}**, karşılaştırılan bölgeler arasında en uygun m² fiyatına ({cheapest_price:,.2f} TL) sahip.")
else:
    diff_pct = ((target_price - cheapest_price) / cheapest_price) * 100
    st.warning(f"⚠️ **Pahalı Seçim:** Hedefiniz olan **{main_district}**, en uygun bölge olan **{cheapest_district}** bölgesine göre %{diff_pct:.1f} daha pahalıdır.")
    st.info(f"💡 **Alternatif:** Yatırım potansiyeli açısından **{cheapest_district}** ({cheapest_price:,.2f} TL/m²) bölgesini değerlendirmenizi öneririm.")

# --- DETAYLI LİSTE ---
with st.expander(f"📂 {main_district} Bölgesindeki Tüm İlanları İncele"):
    st.dataframe(df_global[df_global['District'] == main_district], use_container_width=True)
