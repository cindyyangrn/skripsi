import pickle
import pandas as pd

class TFIDFModel:
    def __init__(self, model_path='vectors/tfidf/tfidf.pkl'):
        """Inisialisasi objek dengan path model TF-IDF yang akan digunakan"""
        self.model_path = model_path
        self.tfidf_vectorizer = None

    def load_model(self):
        """Memuat model TF-IDF yang disimpan"""
        with open(self.model_path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
    
    def transform_texts(self, text_data):
        """Mentransformasi teks menjadi vektor menggunakan model TF-IDF"""
        if self.tfidf_vectorizer is None:
            self.load_model()  # Memuat model jika belum dimuat

        # Transform teks menggunakan TF-IDF yang sudah ada
        tfidf_matrix = self.tfidf_vectorizer.transform(text_data)
        
        # Membuat DataFrame dengan kata sebagai kolom
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())
        
        return tfidf_matrix, tfidf_df
