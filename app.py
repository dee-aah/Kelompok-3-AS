import streamlit as st
import joblib
import pandas as pd
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from streamlit_autorefresh import st_autorefresh

# --- 1. CONFIG & LOAD ---
# Pastikan file pkl sudah ada di folder models/
MODEL_PATH = "models/model_logreg.pkl" 
TFIDF_PATH = "models/tfidf.pkl"
REFRESH_INTERVAL = 5000  # 5 detik

@st.cache_resource
def load_resources():
    try:
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        return model, tfidf
    except:
        st.error("Model atau TFIDF tidak ditemukan. Pastikan sudah upload ke folder models/")
        return None, None

model, tfidf = load_resources()
label_map = {0: "Ham", 1: "Spam", 2: "Toxic"}

# --- 2. SESSION STATE ---
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "next_page_token" not in st.session_state:
    st.session_state.next_page_token = None
if "all_comments" not in st.session_state:
    st.session_state.all_comments = pd.DataFrame(columns=["Waktu", "Komentar", "Prediksi"])
if "start_time" not in st.session_state:
    st.session_state.start_time = None

# --- 3. FUNCTIONS ---
def classify_comment(text):
    if model and tfidf:
        vector = tfidf.transform([text])
        pred = model.predict(vector)[0]
        return label_map[pred]
    return "N/A"

def get_live_chat_id(api_key, video_id):
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.videos().list(
        part="liveStreamingDetails,snippet",
        id=video_id
    )
    response = request.execute()
    items = response.get("items", [])
    if not items:
        return None, None
    
    # Ambil chat ID
    live_chat_id = items[0]["liveStreamingDetails"].get("activeLiveChatId")
    # Jika activeLiveChatId None, coba ambil dari chat yang sudah selesai (Archive)
    # Catatan: YouTube API standar sangat sulit mengambil chat dari video yang sudah selesai.
    # Namun kita bisa memberikan pesan error yang jelas.
    
    return live_chat_id, items[0]["snippet"].get("title")

def fetch_live_chat(api_key, live_chat_id):
    youtube = build("youtube", "v3", developerKey=api_key)
    try:
        request = youtube.liveChatMessages().list(
            liveChatId=live_chat_id,
            part="snippet",
            pageToken=st.session_state.next_page_token
        )
        response = request.execute()
        st.session_state.next_page_token = response.get("nextPageToken")
        return [item["snippet"]["displayMessage"] for item in response.get("items", [])]
    except Exception as e:
        st.warning("Chat mungkin sudah berakhir atau mencapai batas kuota API.")
        return []

def save_comment(text, label):
    new_row = pd.DataFrame({
        "Waktu": [datetime.now()],
        "Komentar": [text],
        "Prediksi": [label]
    })
    st.session_state.all_comments = pd.concat([st.session_state.all_comments, new_row], ignore_index=True)

def highlight(val):
    if val == "Spam": return "background-color:#ffcccc;color:black;font-weight:bold"
    if val == "Toxic": return "background-color:#DC0000;color:white;font-weight:bold"
    return "background-color:#08CB00;color:white;font-weight:bold"

# --- 4. UI STREAMLIT ---
st.set_page_config(page_title="YouTube Spam Detector", layout="wide")
st.title(" YouTube Live :red[Toxic] & :blue[Spam] Detector")

st.sidebar.header("Konfigurasi")
api_key = st.sidebar.text_input("YouTube API Key", type="password")
video_id = st.sidebar.text_input("YouTube Video ID (Contoh: dQw4w9WgXcQ)")
col_start, col_stop = st.sidebar.columns(2)

with col_start:
    if st.button(":green[▶] Mulai", use_container_width=True):
        st.session_state.is_running = True
        st.session_state.start_time = datetime.now() # Set waktu mulai
        st.session_state.all_comments = pd.DataFrame(columns=["Waktu", "Komentar", "Prediksi"]) # Reset data
        st.rerun() # Memastikan UI langsung update

with col_stop:
    if st.button(":red[⏹] Stop", use_container_width=True):
        st.session_state.is_running = False
        st.rerun() # Memastikan UI langsung update

# --- 5. LOGIKA MONITORING ---
if st.session_state.is_running:
    # CEK BATAS 30 MENIT
    time_elapsed = datetime.now() - st.session_state.start_time
    if time_elapsed > timedelta(minutes=20):
        st.session_state.is_running = False
        st.success(" Monitoring selesai: Batas 30 menit tercapai.")
    else:
        st.info(f"⏱ Durasi Berjalan: {str(time_elapsed).split('.')[0]} / 20:00")
        
        if api_key and video_id:
            try:
                live_chat_id, title = get_live_chat_id(api_key, video_id)
                if not live_chat_id:
                    st.error(" Live chat tidak tersedia. Stream mungkin sudah SELESAI atau chat DIMATIKAN.")
                    st.session_state.is_running = False
                else:
                    st.write(f"Monitoring: **{title}**")
                    comments = fetch_live_chat(api_key, live_chat_id)
                    for text in comments:
                        label = classify_comment(text)
                        save_comment(text, label)
            except Exception as e:
                st.error(f"Terjadi kesalahan API: {e}")
        
        st_autorefresh(interval=REFRESH_INTERVAL, key="live_refresh")

# --- 6. DISPLAY METRICS & LOG ---
colA, colB, colC, colD = st.columns(4)
df_display = st.session_state.all_comments
total = len(df_display)
spam = (df_display["Prediksi"] == "Spam").sum()
ham = (df_display["Prediksi"] == "Ham").sum()
toxic = (df_display["Prediksi"] == "Toxic").sum()

colA.metric("Total Komentar", total)
colB.metric("Spam", spam)
colC.metric("Ham", ham)
colD.metric("Toxic", toxic)

st.subheader("Log Komentar Terbaru")
st.dataframe(
    df_display.tail(30).style.applymap(highlight, subset=["Prediksi"]),
    use_container_width=True
)

# --- 7. DOWNLOAD & ANALISIS ---
if not df_display.empty:
    st.download_button(
        " Download Hasil Monitoring (CSV)",
        df_display.to_csv(index=False),
        "hasil_live_monitoring.csv",
        "text/csv"
    )
# --- 8. KESIMPULAN & RINGKASAN (Muncul jika monitoring berhenti) ---
if not st.session_state.is_running and not st.session_state.all_comments.empty:
    st.divider()
    st.subheader(" Kesimpulan Analisis Live Chat")
    
    # Hitung Persentase
    total_chat = len(st.session_state.all_comments)
    spam_count = (st.session_state.all_comments["Prediksi"] == "Spam").sum()
    toxic_count = (st.session_state.all_comments["Prediksi"] == "Toxic").sum()
    ham_count = (st.session_state.all_comments["Prediksi"] == "Ham").sum()
    
    p_spam = (spam_count / total_chat) * 100
    p_toxic = (toxic_count / total_chat) * 100
    
    # Buat Kolom Kesimpulan
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.write("**Statistik Akhir:**")
        st.write(f"- 🟢 Ham (Bersih): {ham_count} ({ham_count/total_chat*100:.1f}%)")
        st.write(f"- 🟡 Spam: {spam_count} ({p_spam:.1f}%)")
        st.write(f"- 🔴 Toxic: {toxic_count} ({p_toxic:.1f}%)")
    
    with c2:
        st.write("**Status Keamanan Chat:**")
        # Logika Penentuan Status
        if p_toxic > 20 or (p_spam + p_toxic) > 40:
            st.error(" KONDISI BAHAYA: Chat didominasi pesan negatif/spam. Disarankan mengaktifkan 'Slow Mode' atau 'Subscribers-only mode' di YouTube.")
        elif p_toxic > 5 or p_spam > 15:
            st.warning(" KONDISI WASPADA: Mulai muncul banyak gangguan. Perlu pengawasan moderator lebih intens.")
        else:
            st.success(" KONDISI AMAN: Interaksi penonton mayoritas positif dan relevan.")

    # Chart Pie untuk Visualisasi
    chart_data = pd.DataFrame({
        "Kategori": ["Ham", "Spam", "Toxic"],
        "Jumlah": [ham_count, spam_count, toxic_count]
    })
    st.bar_chart(chart_data.set_index("Kategori"))