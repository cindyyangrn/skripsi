import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
# import file lain
from preprocessing import Preprocessor
from tfidf import TFIDFModel
from cbow import CBOWModel
from skipgram import SkipgramModel

class App:
    def __init__(self):
        # Inisialisasi variabel
        self.model = None
        self.X = None

    def load_model(self, model_name):
        """Load model sesuai dengan nama model yang diberikan"""
        if model_name == "TF-IDF":
            model = pickle.load(open("models/best_model_svm_tfidf.pkl", "rb"))
        elif model_name == "Word2Vec CBOW":
            model = pickle.load(open("models/best_model_svm_cbow.pkl", "rb"))
        elif model_name == "Word2Vec Skipgram":
            model = pickle.load(open("models/best_model_svm_skipgram.pkl", "rb"))
        return model

    def run(self):
        """Menjalankan aplikasi Streamlit"""
        # Konfigurasi tema warna
        st.set_page_config(page_title="Sweet", page_icon="💐", layout="centered")
        st.markdown("""
            <style>
                .main {background-color: #ffe4e1;}
                .stButton>button {background-color: #ff69b4; color: white;}
                .stTextInput>div>input {border: 2px solid #ff69b4;}
                .stSelectbox>div {border: 2px solid #ff69b4;}
            </style>
            """, unsafe_allow_html=True)

        # Judul aplikasi
        st.title("Analisis Sentimen Komentar Instagram @magangmerdeka")
        st.subheader("Menggunakan TF-IDF dan Word2Vec dengan Metode SVM")

        # Upload file
        uploaded_file = st.file_uploader("Upload file", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            required_columns = ['komentar', 'label']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.warning(f"Kolom {', '.join(missing_columns)} tidak ditemukan dalam file, prediksi sentimen tidak dapat dilakukan.")
            else:
                st.write("Preview file:")
                st.dataframe(df.head())
                
                # Inisialisasi objek Preprocessor
                preprocessor = Preprocessor()
                # Pembersihan data berdasarkan kolom yang dipilih pengguna
                df_cleaned = preprocessor.clean_dataframe(df, 'komentar')

                # Mengupdate kolom 'komentar' dengan hasil preprocessing
                df['komentar'] = df['komentar'].apply(preprocessor.clean_text)

                st.write("Data setelah preprocessing:")
                st.dataframe(df.head())

                # Memilih Ekstraksi Fitur
                extraction_method = st.selectbox(
                    "Pilih Metode Ekstraksi Fitur:", 
                    ["Pilih metode", "TF-IDF", "Word2Vec CBOW", "Word2Vec Skipgram"]
                )

                # Proses berdasarkan metode yang dipilih
                if extraction_method != "Pilih metode":
                    self.model = self.load_model(extraction_method)

                    if extraction_method == "TF-IDF":
                        # Inisialisasi objek TFIDFModel
                        tfidf_handler = TFIDFModel()
                        # Proses transformasi teks pada kolom yang sudah dibersihkan
                        _, tfidf_df = tfidf_handler.transform_texts(df_cleaned['komentar'])  # Gunakan 'komentar' langsung
                        self.X = tfidf_df
                    elif extraction_method == "Word2Vec CBOW":
                        # Menggunakan kelas CBOWModel
                        cbow_model_handler = CBOWModel()
                        # Proses transformasi teks pada kolom yang sudah dibersihkan
                        self.X = cbow_model_handler.transform_texts_to_vectors(df_cleaned['komentar'])  # Gunakan 'komentar' langsung
                    elif extraction_method == "Word2Vec Skipgram":
                        # Inisialisasi objek SkipgramModel
                        skipgram_model_handler = SkipgramModel()
                        # Proses transformasi teks pada kolom yang sudah dibersihkan
                        self.X = skipgram_model_handler.transform_texts_to_vectors(df_cleaned['komentar'])  # Gunakan 'komentar' langsung


                # Klasifikasi Sentimen
                if st.button("Prediksi Sentimen"):
                    if extraction_method == "Pilih metode":
                        st.warning("Silakan pilih metode ekstraksi fitur yang ingin digunakan!")
                    elif self.model is not None and self.X is not None:
                        # Prediksi
                        predictions = self.model.predict(self.X)
                        df['Sentimen'] = predictions

                        df = df.dropna(subset=['label'])
                        true_labels = df['label'].astype(int)

                        predictions = self.model.predict(self.X[:len(df)])
                        df['Sentimen'] = predictions

                        # Hitung akurasi dan matriks konfusi
                        accuracy = accuracy_score(true_labels, predictions)
                        conf_matrix = confusion_matrix(true_labels, predictions)

                        # Tampilkan hasil prediksi sentimen dan akurasi
                        positive_count = sum(df['Sentimen'] == 1)
                        negative_count = sum(df['Sentimen'] == -1)
                        st.write(f"Jumlah prediksi sentimen positif = {positive_count}")
                        st.write(f"Jumlah prediksi sentimen negatif = {negative_count}")
                        st.write(f"**Akurasi model :** {accuracy:.2f}")

                        # Tampilkan hasil prediksi sentimen, akurasi, dan matriks konfusi dalam dua kolom bersebelahan
                        col1, col2 = st.columns(2)

                        with col1:
                            # Tampilkan classification report
                            report = classification_report(true_labels, predictions, target_names=['Negative', 'Positive'], output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.write(report_df)

                        with col2:
                            # Plot matriks konfusi
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="RdPu", ax=ax, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                            ax.set_title('Confusion Matrix')
                            ax.set_xlabel('Predicted Labels')
                            ax.set_ylabel('True Labels')  
                            st.pyplot(fig)

# Menjalankan aplikasi
if __name__ == "__main__":
    app = App()
    app.run()
