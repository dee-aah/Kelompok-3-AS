import joblib
import streamlit as st
import seaborn
import matplotlib

model = joblib.load(r"models\model_logistic_regression.pkl")
tfidf = joblib.load(r"models\tfidf_vectorizer.pkl")

st.title("Aplikasi Klasifikasi Komentar Publik")
st.write("Aplikasi Ini Dibuat dengan Teknologi NLP dengan Memanfaatkan Model Machine Learning Logistic Regression")

input = st.text_input("Masukkan Komentar Anda!!!")
if st.button("Submit"):
    if input.strip() == "":
        st.warning("Komentar Tidak Boleh Kosong")
    else:
        vector =  tfidf.transform([input])
        prediksi = model.predict(vector)[0]

        label_map = {
            0 : "Negatif",
            1 : "Positif"
        }
        st.subheader("Hasil Analisis Komentar")
        st.write("**Komentar : **", label_map.get (prediksi, prediksi))