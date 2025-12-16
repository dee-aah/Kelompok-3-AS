import joblib
import streamlit as st
import seaborn
import matplotlib

model = joblib.load("models/model_svm (2).pkl")
tfidf = joblib.load("models/tfidf.pkl")

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
            0 : "Ham",
            1 : "Spam"
        }
        # 3. Menampilkan Hasil (Menggunakan st.markdown/f-string untuk menghindari potensi sensor)
        hasil_prediksi = label_map.get(prediksi, "Tidak Diketahui")
        
        st.subheader("Hasil Analisis Komentar")
        
        # SOLUSI MASALAH SENSOR/FORMAT: Menggunakan st.markdown dengan f-string
        if hasil_prediksi == "Spam":
            st.error(f"**Klasifikasi:** **{hasil_prediksi}**") # Tampilkan dengan warna error jika Spam
        else:
            st.success(f"**Klasifikasi:** **{hasil_prediksi}**") # Tampilkan dengan warna success jika Ham
        
        st.caption(f"Komentar yang dimasukkan: '{input}'")