import pandas as pd
import streamlit as st
import os
import re
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import datetime

# --- SABİT TANIMLAR ---
CURRENT_YEAR = datetime.now().year

# --- DİL SÖZLÜĞÜ (TAM VE EKSİKSİZ) ---
TRANS = {
    "TR": {
        "title": "ZERO HACK: EMLAK PRO PLATINUM (SAF VERİ)", "upload_label": "Veri Seti (CSV)", "upload_success": "İlan Yüklendi",
        "btn_analyze": "ANALİZİ BAŞLAT", "metric_acc": "AI Doğruluk Skoru (R²)", "metric_total": "Toplam İlan",
        "metric_opp": "Süper Fırsat", "metric_err": "Temizlenen Kayıt", 
        "tab_detail": "📊 DETAYLI ANALİZ", "tab_opp": "💎 FIRSATLAR", "tab_comp": "🎯 OPTIMAL SEÇİM",
        "tab_map": "🗺️ HARİTA", "tab_ai": "🤖 AI ASİSTAN", "col_price": "Fiyat", "col_ai": "AI Değer",
        "col_ref": "2020 Ref.", "col_diff": "Fark", "col_rooms": "Oda Sayısı", "col_age": "Bina Yaşı",
        "col_bath": "Banyo Sayısı", "col_heating": "Isıtma",
        "status_err_low": "⛔ HATALI (SİLİNDİ)", "status_opp": "💎 SÜPER FIRSAT", "status_ok": "✅ PİYASA UYGUN",
        "status_err_high": "❌ PAHALI (SİLİNDİ)", "chat_placeholder": "Veri setine bir soru sor...",
        "ai_unknown": "Anlayamadım. Lütfen kılavuza göz atın.", "loading": "Model eğitiliyor ve veriler işleniyor...",
        "comp_rooms": "Oda Sayısına Göre", "comp_price": "Fiyata Göre", "comp_m2": "m²'ye Göre",
        "comp_select_title": "Karşılaştırma Kriterini Seçin", "comp_list_title": "En Mantıklı {kriter} İlanlar (ROI'ye Göre)",
        "map_title": "Bölgesel Yoğunluk ve Fiyat Haritası", "err_no_file": "Lütfen CSV dosyasını yükleyin.",
        "box_opp_title": "💎 YATIRIMLIK FIRSATLAR", "box_err_title": "🚫 HATA KAYITLARI (GİZLENDİ)",
        "chat_intro": "💡 **İPUCU:** Ben sadece yüklenen dosya hakkındaki soruları yanıtlarım.",
        "cleaning_report": "✅ **VERİ TEMİZLİĞİ:** Başlangıç: {initial}. Atılan (Teknik + Mantıksız + AI Hatalı): {removed}. **Analiz {final} ilanla başladı.**",
        "val_none": "Yok",
        "ai_guide_title": "AI ASİSTAN Soru Kılavuzu",
        "ai_guide_content": """
        Bu veri setine şunları sorabilirsiniz:
        * **Genel Metrikler:** "Toplam kaç ilan var?", "En pahalı ev nerede?", "En ucuz daire hangisi?"
        * **Bölge Bazlı:** "Kadıköy'deki ortalama fiyat ne kadar?", "Esenyurt'ta kaç ilan var?"
        * **Birim Fiyat:** "Ortalama m² fiyatı nedir?", "Ortalama Bina yaşı nedir?"
        """
    },
    "EN": {
        "title": "ZERO HACK: ESTATE PRO PLATINUM (PURE DATA)", "upload_label": "Dataset (CSV)", "upload_success": "Loaded",
        "btn_analyze": "START ANALYSIS", "metric_acc": "AI Accuracy Score (R²)", "metric_total": "Total Listings",
        "metric_opp": "Super Opportunity", "metric_err": "Records Cleared", 
        "tab_detail": "📊 DETAILED ANALYSIS", "tab_opp": "💎 OPPORTUNITIES", "tab_comp": "🎯 OPTIMAL SELECTION", "tab_map": "🗺️ MAP", "tab_ai": "🤖 AI ASSISTANT",
        "col_price": "Price", "col_ai": "AI Value", "col_ref": "2020 Ref.", "col_diff": "Diff",
        "col_rooms": "Rooms", "col_age": "Age", "col_bath": "Bathrooms", "col_heating": "Heating",
        "status_err_low": "⛔ ERROR (DELETED)", "status_opp": "💎 SUPER OPPORTUNITY", "status_ok": "✅ MARKET PRICE",
        "status_err_high": "❌ OVERPRICED (DELETED)", "chat_placeholder": "Ask a question to the dataset...",
        "ai_unknown": "I didn't understand. Please check the guide.", "loading": "Training model...",
        "comp_rooms": "By Number of Rooms", "comp_price": "By Price", "comp_m2": "By m²",
        "comp_select_title": "Select Comparison Criteria",
        "comp_list_title": "Most Optimal {kriter} Listings (By ROI)", "map_title": "Regional Density and Price Map", 
        "err_no_file": "Please upload a CSV file.", "box_opp_title": "💎 INVESTMENT OPPORTUNITIES (CLEAN DATA)", "box_err_title": "🚫 ERROR RECORDS (HIDDEN)",
        "chat_intro": "💡 **TIP:** I only answer questions about the uploaded file.",
        "cleaning_report": "✅ **CLEANING REPORT:** Initial: {initial}. Removed (Impossible/Technical Error): {removed}. **Analysis started with {final} listings.**",
        "val_none": "None",
        "ai_guide_title": "AI ASSISTANT Question Guide",
        "ai_guide_content": """
        You can ask the dataset:
        * **General Metrics:** "How many listings are there?", "Where is the most expensive house?", "What is the cheapest flat?"
        * **District Based:** "What is the average price in Kadikoy?", "How many listings are in Esenyurt?"
        * **Price Per M²:** "What is the average price per m²?", "What is the average building age?"
        """
    },
    "RU": {
        "title": "ZERO HACK: ESTATE PRO PLATINUM (ЧИСТЫЕ ДАННЫЕ)", "upload_label": "CSV Файл", "upload_success": "Загружено",
        "btn_analyze": "НАЧАТЬ АНАЛИЗ", "metric_acc": "Точность ИИ (R²)", "metric_total": "Всего", "metric_opp": "Возможность", 
        "metric_err": "Удалено Записей", "tab_detail": "📊 АНАЛИЗ", "tab_opp": "💎 ВЫГОДНО", "tab_comp": "🎯 ВЫБОР", "tab_map": "🗺️ КАРТА", 
        "tab_ai": "🤖 ИИ", "col_price": "Цена", "col_ai": "ИИ Цена", "col_ref": "2020 Спр.", "col_diff": "Разн.", "col_rooms": "Комнаты", 
        "col_age": "Возраст", "col_bath": "Ванные", "col_heating": "Отопление", "status_err_low": "⛔ ОШИБКА (УДАЛЕНО)", "status_opp": "💎 ВЫГОДНО", 
        "status_ok": "✅ НОРМА", "status_err_high": "❌ ДОРОГО (УДАЛЕНО)", "chat_placeholder": "Спроси данные...", "ai_unknown": "Я не понял. Проверьте гайд.",
        "loading": "Обучение модели...", "comp_rooms": "По кол-ву комнат", "comp_price": "По цене", "comp_m2": "По м²",
        "comp_select_title": "Выберите критерии", "comp_list_title": "Самые выгодные {kriter} объявления", "map_title": "Карта плотности и цен",
        "err_no_file": "Загрузите файл CSV.", "box_opp_title": "💎 ИНВЕСТИЦИИ (ЧИСТЫЕ ДАННЫЕ)", "box_err_title": "🚫 ЗАПИСИ ОШИБОК (СКРЫТЫ)",
        "chat_intro": "💡 **СОВЕТ:** Я отвечаю только на вопросы по загруженному файлу.",
        "cleaning_report": "✅ **ОТЧЕТ:** Начало: {initial}. Удалено (Из-за ошибок): {removed}. **Анализ начат с {final} объявлений.**",
        "val_none": "Нет",
        "ai_guide_title": "Гайд по вопросам к ИИ",
        "ai_guide_content": """
        Вы можете спросить:
        * **Метрики:** "Сколько всего объявлений?", "Где самый дорогой дом?", "Какая самая дешевая квартира?"
        * **По Районам:** "Какова средняя цена в Кадыкей?", "Сколько объявлений в Эсеньюрт?"
        * **Цена за м²:** "Какова средняя цена за м²?", "Какой средний возраст зданий?"
        """
    },
    "AR": {
        "title": "زيرو هاك: إملاك برو بلاتينيوم", "upload_label": "ملف CSV", "upload_success": "تم تحميل الإعلانات",
        "btn_analyze": "ابدأ التحليل", "metric_acc": "دقة AI", "metric_total": "الإجمالي", "metric_opp": "فرصة", 
        "metric_err": "السجلات المطهرة", "tab_detail": "📊 تحليل مفصل", "tab_opp": "💎 الفرص", "tab_comp": "🎯 الاختيار الأمثل", "tab_map": "🗺️ الخريطة", 
        "tab_ai": "🤖 المساعد", "col_price": "السعر", "col_ai": "قيمة AI", "col_ref": "مرجع 2020", "col_diff": "الفرق",
        "col_rooms": "الغرف", "col_age": "العمر", "col_bath": "الحمامات", "col_heating": "التدفئة",
        "status_err_low": "⛔ خطأ (محذوف)", "status_opp": "💎 فرصة", "status_ok": "✅ سعر السوق", "status_err_high": "❌ مرتفع (محذوف)", 
        "chat_placeholder": "اسأل البيانات...", "ai_unknown": "لم أفهم. يرجى مراجعة الدليل.", "loading": "جاري تدريب النموذج...",
        "comp_rooms": "عدد الغرف", "comp_price": "السعر", "comp_m2": "المساحة", "comp_select_title": "اختر معايير المقارنة", 
        "comp_list_title": "أفضل الإعلانات {kriter}", "map_title": "خريطة الكثافة والأسعار", "err_no_file": "يرجى تحميل ملف CSV.", 
        "box_opp_title": "💎 فرص استثمارية (بيانات نظيفة)", "box_err_title": "🚫 سجلات الخطأ (مخفية)", "chat_intro": "💡 **تلميح:** أنا أجيب فقط على الأسئلة المتعلقة بالملف الذي تم تحميله.",
        "cleaning_report": "✅ **تقرير التنظيف:** البداية: {initial}. تم إزالة (بسبب الأخطاء): {removed}. **بدأ التحليل بـ {final} إعلان.**",
        "val_none": "لا يوجد",
        "ai_guide_title": "دليل أسئلة المساعد الآلي",
        "ai_guide_content": """
        يمكنك أن تسأل مجموعة البيانات:
        * **المقاييس العامة:** "كم عدد الإعلانات؟", "أين أغلى منزل؟", "ما هي أرخص شقة؟"
        * **حسب المنطقة:** "ما هو متوسط السعر في كاديكوي؟", "كم عدد الإعلانات في إسنيورت؟"
        * **السعر للمتر المربع:** "ما هو متوسط سعر المتر المربع؟", "ما هو متوسط عمر المبنى؟"
        """
    }
}

# --- REFERANS FİYATLAR (2020) ve İSTANBUL KOORDİNATLARI (Kısaltıldı) ---
REF_PRICES_2020 = { "Esenyurt": 2150, "Sultanbeyli": 2250, "Eyüpsultan": 3400, "Beyoğlu": 5500, "Ataşehir": 5342, "Başakşehir": 4544, "Küçükçekmece": 4076, "Avcılar": 2936, "Beylikdüzü": 2683, "Kadıköy": 8067, "Kağıthane": 4172, "Büyükçekmece": 3456, "Bağcılar": 3255, "Ümraniye": 3767, "Silivri": 2582, "Üsküdar": 5010, "Beşiktaş": 11788, "Bakırköy": 9207, "Sancaktepe": 2613, "Adalar": 6784, "Tuzla": 3262, "Kartal": 3749, "Pendik": 3098, "Çekmeköy": 3088, "Arnavutköy": 2456, "Esenler": 3197, "Şile": 3836, "Şişli": 5592, "Maltepe": 4205, "Sultangazi": 2661, "Zeytinburnu": 4546, "Bayrampaşa": 3701, "Çatalca": 2802, "Bahçelievler": 3353, "Gaziosmanpaşa": 3106, "Sarıyer": 10589, "Fatih": 4202, "Güngören": 3206 }
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
    "Ümraniye": [41.0256, 29.0963], "Üsküdar": [41.0261, 29.0152], "Zeytinburnu": [40.9904, 28.9001],
    "Çatalca": [41.1432, 28.4593], "Bahçelievler": [41.0001, 28.8601], "Gaziosmanpaşa": [41.0581, 28.9124],
    "Fatih": [41.0102, 28.9403], "Güngören": [41.0253, 28.8651]
}

# --- TÜM ÖZELLİK KELİMELERİ (Kullanım kolaylığı için kısaltıldı) ---
ALL_KEYWORDS = ['bölge', 'mahalle', 'semt', 'ilan tarihi', 'kat konumu', 'kat sayısı', 'mobilyalı', 'kullanım durumu', 'kiralamaya uygun', 
    'kimden', 'takas', 'batı cephe', 'doğu cephe', 'güney cephe', 'kuzey cephe', 'adsl', 'ahşap doğrama', 'akıllı ev', 
    'alarmı (hırsız)', 'alarm (yangın)', 'alaturka tuvalet', 'alüminyum doğrama', 'amerikan kapı', 'amerikan mutfak', 
    'ankastre', 'asansör', 'barbekü', 'ev aletleri', 'boyalı', 'bulaşık makinesi', 'buzdolabı', 'duvar kağıdı', 'duş', 
    'ebeveyn banyosu', 'fiber internet', 'giyinme odası', 'dolap', 'görüntülü interkom', 'hilton banyosu', 'interkom sistemi', 
    'yalıtımlı cam', 'jakuzi', 'alçıpan', 'bodrum', 'klima', 'küvet', 'laminat parke', 'marley mobilya', 'ankastre mutfak', 
    'laminat mutfak', 'doğalgazlı mutfak', 'pvc doğrama', 'jaluzi', 'parke zemin', 'seramik zemin', 'set üstü ocak', 
    'spot aydınlatma', 'teras', 'termosifon', 'vestiyer', 'wi-fi', 'yüz tanıma ve parmak i̇zi', 'çamaşır makinesi', 
    'çamaşırhane', 'çelik kapı', 'su ısıtıcı', 'şömine', 'buhar odası', 'güvenlik banyosu', 'güçlendirici', 'ısı yalıtımı', 
    'jeneratör', 'kablo tv', 'kapalı garaj', 'kapıcı', 'kreş', 'özel havuzlu', 'otopark', 'oyun alanı', 'sauna', 
    'ses yalıtımı', 'dış cephe kaplaması', 'spor alanı', 'su deposu', 'tenis kortu', 'uydu', 'yangın merdiveni', 
    'açık yüzme havuzu', 'kapalı yüzme havuzu', 'geniş koridor', 'giriş / rampa', 'merdivenler', 'oda kapısı', 'priz / elektrik anahtarı', 
    'kapı kolu / korkuluk', 'tuvalet', 'yüzme havuzu', 'alışveriş merkezi', 'belediye', 'cami', 'cemevi', 'sahile yakın', 
    'eczane', 'eğlence merkezi', 'fuar', 'hastane', 'sinagog', 'kilise', 'lise', 'market', 'park', 'polis karakolu', 
    'sağlık kliniği', 'ilçe marketi', 'spor salonu', 'üniversite', 'ilkokul-ortaokul', 'itfaiye', 'şehir merkezi', 
    'otoyol', 'avrasya tüneli', 'boğaz köprüleri', 'cadde', 'deniz otobüsü', 'dolu', 'e-5', 'havaalanı', 'marmaray', 
    'metro', 'metrobüs', 'minibüs', 'otobüs durağı', 'sahil', 'tem', 'teleferik', 'tramvay', 'tren i̇stasyonu', 
    'troleybüs', 'iskele', 'boğaz denizi', 'doğa', 'göl', 'havuz', 'park ve yeşil alan', 'şehir', 'asma kat', 
    'ara kat dubleks', 'bahçe dubleks', 'bahçe katı', 'bahçe üst kat', 'garaj / dükkan', 'üst giriş katı', 'kat dubleks', 
    'özel giriş', 'rerse dubleks', 'tripleks', 'zemin kat', 'çatı dubleks', 'teslim alma zamanı'
]

# --- VERİ YÜKLEME (Temel ve Teknik Temizlik) ---
@st.cache_data
def load_data(file_path):
    try:
        if isinstance(file_path, str):
            try: df = pd.read_csv(file_path, sep=None, encoding='utf-8', engine='python')
            except: df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        else:
            df = pd.read_csv(file_path, sep=None, encoding='utf-8', engine='python')

        df.columns = df.columns.str.strip()
        
        col_map = {}
        all_cols = list(df.columns)
        
        def find_col(keywords, target_name, mandatory=False):
            for col in all_cols:
                if any(k.lower() in col.lower() for k in keywords):
                    col_map[col] = target_name
                    return col
            if mandatory: return None
            return 'not_found'

        # Kritik Sütunların Eşlenmesi
        if not find_col(['district', 'ilçe'], 'District', mandatory=True): return None, "Kritik 'İlçe' sütunu eksik.", None
        if not find_col(['price', 'fiyat'], 'Price', mandatory=True): return None, "Kritik 'Fiyat' sütunu eksik.", None
        if not (find_col(['m² (net)', 'net m2', 'net'], 'm²') or find_col(['m² (brüt)', 'gross'], 'm²')): return None, "Kritik 'm²' sütunu eksik.", None
        
        find_col(['neighborhood', 'mahalle'], 'Neighborhood')
        find_col(['oda', 'room'], 'Oda_Text')
        find_col(['bina yaşı', 'building age', 'age'], 'Bina_Yasi')
        find_col(['banyo sayısı', 'number of bathrooms', 'bath'], 'Banyo_Sayisi')
        find_col(['ısıtma', 'heating'], 'Isitma')
        find_col(['kat konumu'], 'Kat_Konumu') # Mantıksal kontrol
        find_col(['kat sayısı'], 'Kat_Sayisi') # Mantıksal kontrol
        find_col(['teras'], 'Teras') # Mantıksal kontrol
        
        # Coğrafi Mantık Kontrolü için Sütunlar
        find_col(['sahile yakın', 'deniz kenarı'], 'Sahile_Yakin')
        find_col(['göl'], 'Göl')
        find_col(['yüzme havuzu', 'havuz', 'özel havuzlu'], 'Yuzme_Havuzu')
        find_col(['boğaz denizi', 'boğaz köprüleri', 'deniz'], 'Bogaz_Deniz')
        find_col(['doğa', 'park ve yeşil alan'], 'Doga_Yesil_Alan') 

        df = df.rename(columns=col_map)

        def clean_num(val):
            if isinstance(val, str): 
                val = re.sub(r'[^\d.]', '', val.replace('TL', '').replace('.', '').replace(',', '.'))
            try: return float(val)
            except: return np.nan

        df['Price'] = df['Price'].apply(clean_num)
        df['m²'] = df['m²'].apply(clean_num)
        
        if 'Banyo_Sayisi' in df.columns:
            df['Banyo_Sayisi'] = df['Banyo_Sayisi'].apply(lambda x: clean_num(x) if pd.notna(x) else 0)
        
        if 'Bina_Yasi' in df.columns:
            df['Bina_Yasi'] = df['Bina_Yasi'].apply(lambda x: clean_num(x) if pd.notna(x) else 0)
        if 'Kat_Sayisi' in df.columns:
            df['Kat_Sayisi'] = df['Kat_Sayisi'].apply(lambda x: clean_num(x) if pd.notna(x) else 0)

        initial_count = len(df)
        df_cleaned = df.copy()
        
        df_cleaned = df_cleaned[df_cleaned.get('Bina_Yasi', 0).apply(lambda x: x <= 125 and x >= 0)] 
        
        df_cleaned = df_cleaned.dropna(subset=['Price', 'm²'])
        df_cleaned = df_cleaned[(df_cleaned['Price'] > 0) & (df_cleaned['m²'] > 0) & (df_cleaned['m²'] < 1000)]
        
        removed_count = initial_count - len(df_cleaned)
        st.session_state['cleaning_report'] = {"initial": initial_count, "removed": removed_count, "final": len(df_cleaned)}

        df = df_cleaned 

        df['District'] = df['District'].str.title().str.strip()
        
        def parse_room(val):
            try:
                nums = re.findall(r'\d+', str(val))
                return int(nums[0]) if nums else 2
            except: return 2
        
        if 'Oda_Text' in df.columns: df['Oda_Sayisi'] = df['Oda_Text'].apply(parse_room)
        else: df['Oda_Sayisi'] = 2

        # Feature kolonlarını boolean'a çevir
        feature_cols = []
        for col_name in df.columns: 
            if col_name not in ['District', 'Price', 'm²', 'Oda_Text', 'Bina_Yasi', 'Banyo_Sayisi', 'Isitma', 'Oda_Sayisi', 'Neighborhood', 'Kat_Konumu', 'Kat_Sayisi']:
                if df[col_name].dtype in [np.int64, np.float64, bool] and df[col_name].nunique() <= 2 and df[col_name].max() <= 1:
                    df[col_name] = df[col_name].fillna(0).apply(lambda x: 1 if x > 0 else 0)
                    if col_name not in feature_cols:
                        feature_cols.append(col_name)
                        
        return df, None, feature_cols

    except Exception as e: return None, str(e), None

# --- Yapay Zeka Asistanı Fonksiyonu (Global Kapsamda) ---
def smart_data_assistant_multilang(df, query, lang_code):
    query = query.lower()
    k = TRANS.get(lang_code, TRANS["TR"])
    
    if any(x in query for x in ["toplam", "total", "всего", "مجموع", "kaç ilan", "how many"]):
        return k["ai_total_resp"].format(count=len(df))
    
    if any(x in query for x in ["pahalı", "expensive", "дорогой", "أغلى"]):
        row = df.sort_values(by='Price', ascending=False).iloc[0]
        return f"**{row['District']}** ({row['m²']} m²): {row['Price']:,.0f} TL"
    
    if any(x in query for x in ["ucuz", "cheap", "дешевый", "أرخص"]):
        temp_df = df[df['Price'] > 10000].sort_values(by='Price', ascending=True)
        if not temp_df.empty:
            row = temp_df.iloc[0]
            return f"**{row['District']}** ({row['m²']} m²): {row['Price']:,.0f} TL"
        else:
            return "Çok ucuz mantıklı ilan bulunamadı."
    
    if any(x in query for x in ["ortalama m²", "average m2 price", "средняя цена м²", "متوسط سعر المتر"]):
        avg_price_m2 = (df['Price'] / df['m²']).mean()
        return f"Ortalama m² fiyatı: **{avg_price_m2:,.0f} TL**"

    if any(x in query for x in ["ortalama yaş", "average age", "средний возраст", "متوسط عمر المبنى"]) and 'Bina_Yasi' in df.columns:
        avg_age = df['Bina_Yasi'].mean()
        return f"Ortalama Bina Yaşı: **{avg_age:,.1f}**"
    
    districts = df['District'].unique()
    for d in districts:
        if d.lower() in query:
            dist_df = df[df['District'] == d]
            avg_price = dist_df['Price'].mean()
            count = len(dist_df)
            return f"📍 **{d}** Analizi:\n- İlan Sayısı: {count}\n- Ort. Fiyat: **{avg_price:,.0f} TL**"

    return k["ai_unknown"]

# --- MANTIK KONTROL FONKSİYONU ---
def check_logical_inconsistencies(df):
    """
    Mantıksal olarak imkânsız / absürt ilanları otomatik tespit eder.
    True dönen satırlar HATALI kabul edilir ve analizden silinir.
    """

    mask = pd.Series(False, index=df.index)

    # --- DENİZ / GÖL / HAVUZ / DOĞA ÇAKIŞMASI ---
    cols = []
    for c in ['Bogaz_Deniz', 'Göl', 'Yuzme_Havuzu', 'Doga_Yesil_Alan']:
        if c in df.columns:
            cols.append(c)

    if len(cols) >= 2:
        mask |= (df[cols].sum(axis=1) >= 3)

    # --- DENİZ + GÖL (KESİN HATA) ---
    if 'Bogaz_Deniz' in df.columns and 'Göl' in df.columns:
        mask |= (df['Bogaz_Deniz'] == 1) & (df['Göl'] == 1)

    # --- TERAS + ZEMİN / BAHÇE KAT ---
    if 'Kat_Konumu' in df.columns and 'Teras' in df.columns:
        kat = df['Kat_Konumu'].astype(str).str.lower()
        mask |= (df['Teras'] == 1) & (kat.str.contains('zemin|bahçe|giriş', na=False))

    # --- YÜKSEK BİNA + ZEMİN KAT ---
    if 'Kat_Sayisi' in df.columns and 'Kat_Konumu' in df.columns:
        kat = df['Kat_Konumu'].astype(str).str.lower()
        mask |= (df['Kat_Sayisi'] >= 8) & (kat.str.contains('zemin|bahçe', na=False))

    # --- AŞIRI m² (DAİRE İÇİN) ---
    mask |= (df['m²'] < 15) | (df['m²'] > 800)

    # --- AŞIRI ESKİ BİNA ---
    if 'Bina_Yasi' in df.columns:
        mask |= (df['Bina_Yasi'] > 150)

    # --- FİYAT / m² ABSÜRTLÜĞÜ ---
    birim_fiyat = df['Price'] / df['m²']
    mask |= (birim_fiyat < 1000) | (birim_fiyat > 500_000)

    return mask


def train_model_and_compare(df_raw_for_train, feature_cols, lang_code):
    
    df = df_raw_for_train.copy()
    
    # --- 1. ABSÜRT MANTIK HATALARINI SİLME ---
    logical_errors = check_logical_inconsistencies(df)
    
    df_absurd_removed = df[~logical_errors].copy()
    
    removed_absurd = len(df) - len(df_absurd_removed)
    df = df_absurd_removed 
    
    initial_for_ai_check = len(df) 
    
    # --- 2. AI MODELİNİ EĞİTME VE TAHMİN ---
    
    df_train = df.copy() 
    
    # Uç Değerleri Atma (Fiyat/m² outlier)
    df_train['Birim_Fiyat'] = df_train['Price'] / df['m²'] 
    Q1 = df_train['Birim_Fiyat'].quantile(0.05) 
    Q3 = df_train['Birim_Fiyat'].quantile(0.95)
    df_train = df_train[(df_train['Birim_Fiyat'] >= Q1) & (df_train['Birim_Fiyat'] <= Q3)]

    # Target Encoding (Konum Skoru)
    if 'Neighborhood' in df.columns:
        neigh_map = df_train.groupby('Neighborhood').apply(lambda x: (x['Price']/x['m²']).median()).to_dict()
        df['Konum_Skoru'] = df['Neighborhood'].map(neigh_map).fillna(df['Price'].median()/df['m²'].median())
        df_train['Konum_Skoru'] = df['Neighborhood'].map(neigh_map)
    else:
        dist_map = df_train.groupby('District').apply(lambda x: (x['Price']/x['m²']).median()).to_dict()
        df['Konum_Skoru'] = df['District'].map(dist_map)
        df_train['Konum_Skoru'] = df['District'].map(dist_map)

    features = ['m²', 'Oda_Sayisi', 'Konum_Skoru'] + feature_cols
    if 'Bina_Yasi' in df.columns: features.append('Bina_Yasi')
    if 'Banyo_Sayisi' in df.columns: features.append('Banyo_Sayisi')

    X_train_data = df_train[features].fillna(0)
    y_train_data = np.log1p(df_train['Price']) 
    
    X_train, X_test, y_train, y_test = train_test_split(X_train_data, y_train_data, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(np.expm1(y_test), np.expm1(y_pred))

    # Tüm Dataya Uygula
    X_full = df[features].fillna(0)
    df['AI_Tahmin'] = np.expm1(model.predict(X_full))
    
    # 2020 Referans
    def get_ref_price(district, m2):
        for key, val in REF_PRICES_2020.items():
            if key.lower() in district.lower(): return val * m2
        return None

    df['Ref_2020_Deger'] = df.apply(lambda row: get_ref_price(row['District'], row['m²']), axis=1)
    
    # Durum Belirleme (Sadece Fiyat Sapmasını Kontrol Eder)
    def determine_status(row):
        target = row['Ref_2020_Deger'] if pd.notnull(row['Ref_2020_Deger']) else row['AI_Tahmin']
        if target == 0 or pd.isna(target): return "N/A"
        
        diff = ((row['Price'] - target) / target) * 100
        
        current_trans = TRANS[lang_code]
        if diff < -40: return current_trans["status_err_low"] # ÇOK Ucuz Fiyat Hatası (Silinecek)
        if -40 <= diff < -15: return current_trans["status_opp"]
        if -15 <= diff <= 25: return current_trans["status_ok"]
        return current_trans["status_err_high"] # ÇOK Pahalı Fiyat Hatası (Silinecek)

    df['Durum'] = df.apply(determine_status, axis=1)
    df['Sapma_Genel_%'] = df.apply(lambda row: ((row['Price'] - (row['Ref_2020_Deger'] if pd.notnull(row['Ref_2020_Deger']) else row['AI_Tahmin'])) / (row['Ref_2020_Deger'] if pd.notnull(row['Ref_2020_Deger']) else row['AI_Tahmin'])) * 100, axis=1)

    # --- 3. AI Fiyat Hatalarını SİLME (Nihai Temizlik) ---
    valid_statuses = [TRANS[lang_code]["status_opp"], TRANS[lang_code]["status_ok"]]
    
    df_final = df[df['Durum'].isin(valid_statuses)].copy() 
    
    removed_ai_outliers = initial_for_ai_check - len(df_final)

    # TEMİZLİK RAPORUNU GÜNCELLE
    initial_tech_removed = st.session_state['cleaning_report']['removed']
    total_removed = initial_tech_removed + removed_absurd + removed_ai_outliers
    
    st.session_state['cleaning_report']['removed'] = total_removed
    st.session_state['cleaning_report']['final'] = len(df_final)
    
    return df_final, r2

# --- ARAYÜZ BAŞLANGIÇ ---
selected_lang_code = st.session_state.get('selected_lang', 'TR')
T = TRANS[selected_lang_code]

# CSS / RTL Ayarları (Kısaltıldı)
direction = "rtl" if selected_lang_code == "AR" else "ltr"
align = "right" if selected_lang_code == "AR" else "left"

st.markdown(f"""
<style>
    .stApp {{ direction: {direction}; }}
    h1, h2, h3 {{ text-align: {align}; }}
    .user-msg, .ai-msg {{ text-align: {align}; direction: {direction}; }}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1040/1040993.png", width=80)
    st.title(T["title"])
    
    new_lang = st.selectbox("Language / Dil / Язык / اللغة", ["TR", "EN", "RU", "AR"], index=["TR", "EN", "RU", "AR"].index(selected_lang_code))
    if new_lang != selected_lang_code:
        st.session_state['selected_lang'] = new_lang
        st.rerun()
    
    uploaded_file = st.file_uploader(T["upload_label"], type=['csv'])
    if not uploaded_file and os.path.exists("veri.csv"): uploaded_file = "veri.csv"

    df_raw = None
    err = None
    feats = None
    if uploaded_file:
        df_raw, err, feats = load_data(uploaded_file)
    
    if err:
        st.error(err)
        st.stop()
        
    if df_raw is None:
        st.info(T["err_no_file"])
        st.stop()
        
    st.success(f"✅ {len(df_raw):,} {T['upload_success']}")
    
    if st.button(T["btn_analyze"], type="primary"):
        with st.spinner(T["loading"]):
            df_res, score = train_model_and_compare(df_raw, feats, new_lang)
            st.session_state['data'] = df_res
            st.session_state['score'] = score
            st.session_state['feats'] = feats
    


# --- ANA EKRAN ---
if 'data' in st.session_state:
    df = st.session_state['data']
    r2 = st.session_state['score']
    
    # Tüm Sütunları Yakalama
    IGNORE_COLS = ['Oda_Text', 'Neighborhood', 'Konum_Skoru', 'Birim_Fiyat', 'Kat_Konumu_Str'] 
    ALL_DISPLAY_COLS = [col for col in df.columns if col not in IGNORE_COLS]
    
    # Çeviri Sözlüğü oluştur
    col_config = {}
    for col in ALL_DISPLAY_COLS:
        if col == 'Price': col_config[col] = st.column_config.NumberColumn(T["col_price"], format="%d TL")
        elif col == 'AI_Tahmin': col_config[col] = st.column_config.NumberColumn(T["col_ai"], format="%d TL")
        elif col == 'Ref_2020_Deger': col_config[col] = st.column_config.NumberColumn(T["col_ref"], format="%d TL")
        elif col == 'Sapma_Genel_%': col_config[col] = st.column_config.NumberColumn(T["col_diff"], format="%.1f%%")
        elif col == 'Oda_Sayisi': col_config[col] = T["col_rooms"]
        elif col == 'Banyo_Sayisi': col_config[col] = T["col_bath"]
        elif col == 'Bina_Yasi': col_config[col] = T["col_age"]
        elif col == 'Isitma': col_config[col] = T["col_heating"]
        elif df[col].dtype in [np.int64, np.float64, bool] and col not in ['Price', 'm²']:
             df[col] = df[col].apply(lambda x: T["val_none"] if (x == 0 or x is False) else x)
             if col not in col_config:
                 col_config[col] = col
        elif col not in col_config:
             col_config[col] = col

    # METRİK KARTLARI
    c1, c2, c3, c4 = st.columns(4)
    opp_count = len(df[df['Durum'] == T["status_opp"]])
    final_count = st.session_state['cleaning_report']['final']
    initial_count = st.session_state['cleaning_report']['initial']
    removed_count = st.session_state['cleaning_report']['removed']
    
    r2_color = "#00e676" if r2 > 0.85 else ("#ffab00" if r2 > 0.75 else "#ff1744")
    
    with c1: st.markdown(f'<div class="metric-container" style="border-color:{r2_color}"><div class="metric-value" style="color:{r2_color}">%{r2*100:.1f}</div><div class="metric-label">{T["metric_acc"]}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-container"><div class="metric-value">{final_count:,}</div><div class="metric-label">{T["metric_total"]}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-container" style="border-color:#00e676"><div class="metric-value" style="color:#00e676">{opp_count:,}</div><div class="metric-label">{T["metric_opp"]}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-container" style="border-color:#ff1744"><div class="metric-value" style="color:#ff1744">{removed_count:,}</div><div class="metric-label">{T["metric_err"]}</div></div>', unsafe_allow_html=True)

    # Temizlik Raporu
    st.info(f"**{T['cleaning_report'].format(initial=initial_count, removed=removed_count, final=final_count)}**")
    
    st.write("")
    
    # SEKMELER
    tabs = st.tabs([T["tab_detail"], T["tab_opp"], T["tab_comp"], T["tab_map"], T["tab_ai"]])

    # 1. DETAYLI ANALİZ (Tüm Sütunlar Kaydırılabilir)
    with tabs[0]:
        st.markdown(f"### 📊 {T['tab_detail']}")
        
        st.dataframe(df[ALL_DISPLAY_COLS], column_config=col_config, use_container_width=True, height=600)

    # 2. FIRSATLAR
    with tabs[1]:
        st.success(T["box_opp_title"])
        opps = df[df['Durum'] == T["status_opp"]].sort_values(by='Sapma_Genel_%', ascending=False)
        st.dataframe(opps[ALL_DISPLAY_COLS], column_config=col_config, use_container_width=True)
    
    # 3. OPTIMAL SEÇİM (KARŞILAŞTIRMA)
    with tabs[2]:
        st.markdown(f"### 🎯 {T['tab_comp']}")
        
        clean_df = df.copy() # Zaten temizlenmiş
        
        comparison_options = {
            T["comp_rooms"]: "Oda Sayısına Göre", 
            T["comp_price"]: "Fiyata Göre", 
            T["comp_m2"]: "m²'ye Göre"
        }
        selected_option_key = st.selectbox(T["comp_select_title"], list(comparison_options.keys()))
        comparison_type = comparison_options[selected_option_key]

        if comparison_type == "Oda Sayısına Göre":
            kriter_name = selected_option_key
            st.markdown(f"#### {T['comp_list_title'].format(kriter=kriter_name)}")
            
            best_by_room = clean_df.loc[clean_df.groupby('Oda_Sayisi')['Sapma_Genel_%'].idxmax()].sort_values(by='Sapma_Genel_%', ascending=False)
            
            st.dataframe(best_by_room[ALL_DISPLAY_COLS], column_config=col_config, use_container_width=True)
            
        elif comparison_type == "Fiyata Göre":
            kriter_name = selected_option_key
            st.markdown(f"#### {T['comp_list_title'].format(kriter=kriter_name)}")
            
            best_by_roi = clean_df.sort_values(by='Sapma_Genel_%', ascending=False).head(10)
            st.dataframe(best_by_roi[ALL_DISPLAY_COLS], column_config=col_config, use_container_width=True)

        elif comparison_type == "m²'ye Göre":
            kriter_name = selected_option_key
            st.markdown(f"#### {T['comp_list_title'].format(kriter=kriter_name)}")
            
            best_by_m2_value = clean_df.sort_values(by='Sapma_Genel_%', ascending=False).head(10)
            st.dataframe(best_by_m2_value[ALL_DISPLAY_COLS], column_config=col_config, use_container_width=True)


    # 4. HARİTA (Tüm Bölgeler Görünür)
    with tabs[3]:
        st.markdown(f"### {T['map_title']}")
        try:
            m = folium.Map(location=[41.0082, 28.9784], zoom_start=9, tiles="CartoDB dark_matter")
            dist_summary = df.groupby('District').agg({'Price': 'mean', 'm²': 'count', 'Sapma_Genel_%': 'mean'}).reset_index()
            
            for idx, row in dist_summary.iterrows():
                coords = ISTANBUL_COORDS.get(row['District'])
                
                if coords:
                    color = "#00e676" if row['Sapma_Genel_%'] < -5 else ("#ff1744" if row['Sapma_Genel_%'] > 10 else "#29b6f6")
                    
                    folium.CircleMarker(
                        location=coords, 
                        radius=5 + (row['m²'] / dist_summary['m²'].max() * 20),
                        popup=f"<b>{row['District']}</b><br>{T['col_price']}: {row['Price']:,.0f} TL",
                        color=color, 
                        fill=True, 
                        fill_color=color, 
                        fill_opacity=0.6
                    ).add_to(m)

            st_folium(m, width="100%", height=500)
        except Exception as e: st.error(f"Harita hatası: {e}")

    # 5. AI ASİSTAN (Kılavuz İçinde)
    with tabs[4]:
        st.markdown(f"### {T['tab_ai']}")
        chat_col, info_col = st.columns([3, 1])
        
        with info_col:
            st.markdown(f"#### {T['ai_guide_title']}")
            st.info(T['ai_guide_content']) # Kılavuz buraya taşındı.
        
        with chat_col:
            chat_cont = st.container(height=400, border=True)
            if "messages" not in st.session_state: st.session_state.messages = []
            
            for msg in st.session_state.messages:
                cls = "user-msg" if msg["role"] == "user" else "ai-msg"
                chat_cont.markdown(f"<div class='{cls}'>{msg['content']}</div>", unsafe_allow_html=True)
                
            if prompt := st.chat_input(T["chat_placeholder"]):
                st.session_state.messages.append({"role": "user", "content": prompt})
                chat_cont.markdown(f"<div class='user-msg'>{prompt}</div>", unsafe_allow_html=True)
                
                response = smart_data_assistant_multilang(df, prompt, selected_lang_code)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                chat_cont.markdown(f"<div class='ai-msg'>{response}</div>", unsafe_allow_html=True)
