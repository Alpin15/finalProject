import streamlit as st
import streamlit.components.v1 as stc
import pickle
import numpy as np

@st.cache_resource
def load_components():
    with open('xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vektorizer.pkl', 'rb') as f:
        vektorizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, vektorizer, le

model, vektorizer, le = load_components()

def predict_with_prob(text):
    X_input = vektorizer.transform([text])
    probs = model.predict_proba(X_input)[0]
    predicted_label = model.predict(X_input)[0]
    predicted_class = le.inverse_transform([predicted_label])[0]
    return predicted_class, probs.max()

html_temp = """<div style="background-color:#000;padding:10px;border-radius:10px">
                <h1 style="color:#fff;text-align:center">Fake News Prediction</h1> 
                <h4 style="color:#fff;text-align:center">Deteksi berita palsu dengan AI</h4> 
                """

desc_temp = """ ### Fake News Prediction App 
                Aplikasi ini digunakan untuk mendeteksi apakah suatu berita atau pernyataan adalah palsu atau tidak.
                
                #### Data Source
                Kaggle: <Link dataset>
                """

def main():
    stc.html(html_temp)
    menu = ["Home", "Fake News"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Fake News":
        run_ml_app()

def run_ml_app():
    st.markdown("""
        <div style="padding:10px;border-radius:10px;margin-bottom:20px;">
            <h2 style="color:#000;">Masukkan berita atau pernyataan (dalam Bahasa Inggris):</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Input teks
    news_text = st.text_area("", placeholder="Masukkan teks berita di sini...", height=150)
    
    # Tombol deteksi
    if st.button("Deteksi", key="predict_button"):
        if news_text.strip() == "":
            st.warning("Masukkan teks berita terlebih dahulu!")
        else:
            # Prediksi
            predicted_class, confidence = predict_with_prob(news_text)
            
            # Tampilkan hasil
            st.markdown("---")
            st.subheader("Hasil Deteksi:")
            
            if predicted_class == "Fake":
                st.error(f"Prediksi: {predicted_class} (Probabilitas: {confidence:.2f})")
            else:
                st.success(f"Prediksi: {predicted_class} (Probabilitas: {confidence:.2f})")

if __name__ == "__main__":
    main()