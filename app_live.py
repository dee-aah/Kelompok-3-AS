import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
from googleapiclient.discovery import build
from streamlit_autorefresh import st_autorefresh


MODEL_PATH = "models/model_svm (2).pkl"
TFIDF_PATH = "models/tfidf.pkl"
REFRESH_INTERVAL = 5000  # ms

@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    return model, tfidf

model, tfidf = load_resources()
label_map = {0: "Ham", 1: "Spam"}

if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "next_page_token" not in st.session_state:
    st.session_state.next_page_token = None
if "all_comments" not in st.session_state:
    st.session_state.all_comments = pd.DataFrame(
        columns=["Waktu", "Komentar", "Prediksi"]
    )

def classify_comment(text):
    vector = tfidf.transform([text])
    pred = model.predict(vector)[0]
    return label_map[pred]


def get_live_chat_id(api_key, video_id):
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.videos().list(
        part="liveStreamingDetails",
        id=video_id
    )
    response = request.execute()
    items = response.get("items", [])
    if not items:
        return None
    return items[0]["liveStreamingDetails"].get("activeLiveChatId")


def fetch_live_chat(api_key, live_chat_id):
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.liveChatMessages().list(
        liveChatId=live_chat_id,
        part="snippet",
        pageToken=st.session_state.next_page_token
    )
    response = request.execute()
    st.session_state.next_page_token = response.get("nextPageToken")
    return [item["snippet"]["displayMessage"] for item in response.get("items", [])]


def save_comment(text, label):
    new_row = pd.DataFrame({
        "Waktu": [datetime.now()],
        "Komentar": [text],
        "Prediksi": [label]
    })
    st.session_state.all_comments = pd.concat(
        [st.session_state.all_comments, new_row],
        ignore_index=True
    )


def highlight(val):
    if val == "Spam":
        return "background-color:#ffcccc;font-weight:bold"
    return "background-color:#ccffcc;font-weight:bold"

st.set_page_config(page_title="YouTube Spam Detector", layout="wide")
st.title("YouTube Live & Post Live Spam Detector")
st.write("Deteksi **Spam / Ham** dari live chat YouTube dan tetap bisa dianalisis setelah live selesai.")


st.sidebar.header("Konfigurasi")
api_key = st.sidebar.text_input("YouTube API Key", type="password")
video_id = st.sidebar.text_input("YouTube Video ID")

col1, col2 = st.sidebar.columns(2)
if col1.button("▶ Mulai"):
    st.session_state.is_running = True
if col2.button("⏹ Stop"):
    st.session_state.is_running = False

if st.session_state.is_running:
    if api_key and video_id:
        try:
            live_chat_id = get_live_chat_id(api_key, video_id)
            if not live_chat_id:
                st.error("Live chat tidak tersedia (live belum mulai / chat dimatikan)")
            else:
                comments = fetch_live_chat(api_key, live_chat_id)
                for text in comments:
                    label = classify_comment(text)
                    save_comment(text, label)
        except Exception as e:
            st.error(f"Gagal mengambil live chat: {e}")
    else:
        st.warning("Masukkan API Key dan Video ID")

    st_autorefresh(interval=REFRESH_INTERVAL, key="live_refresh")

colA, colB, colC = st.columns(3)
total = len(st.session_state.all_comments)
spam = (st.session_state.all_comments["Prediksi"] == "Spam").sum()
ham = (st.session_state.all_comments["Prediksi"] == "Ham").sum()

colA.metric("Total Komentar", total)
colB.metric("Spam", spam)
colC.metric("Ham", ham)

st.subheader("Log Komentar")
st.dataframe(
    st.session_state.all_comments.tail(30).style.applymap(
        highlight, subset=["Prediksi"]
    ),
    use_container_width=True
)


st.subheader("Analisis Setelah Live")
if not st.session_state.all_comments.empty:
    df = st.session_state.all_comments.copy()
    df["Menit"] = pd.to_datetime(df["Waktu"]).dt.floor("min")
    spam_per_minute = df[df["Prediksi"] == "Spam"].groupby("Menit").size()

    st.line_chart(spam_per_minute)

    st.download_button(
        " Download CSV",
        df.to_csv(index=False),
        "hasil_livechat.csv"
    )
else:
    st.info("Belum ada data komentar")
