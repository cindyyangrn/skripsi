from gensim.models import Word2Vec
import numpy as np

class SkipgramModel:
    def __init__(self, model_filepath='vectors/skipgram/skipgram.model', vector_size=150):
        """Inisialisasi dengan memuat model Skipgram dan ukuran vektor"""
        self.model = Word2Vec.load(model_filepath)
        self.vector_size = vector_size

    def get_average_vector(self, tokens):
        """Menghitung rata-rata vektor dari tokens yang diberikan"""
        vecs = [self.model.wv[word] for word in tokens if word in self.model.wv]
        if len(vecs) > 0:
            return np.mean(vecs, axis=0)
        else:
            return np.zeros(self.vector_size)

    def transform_texts_to_vectors(self, text_data):
        """Mengonversi dataset teks menjadi vektor rata-rata menggunakan model Word2Vec"""
        vectors = np.array([self.get_average_vector(text.split()) for text in text_data])
        return vectors  # Mengembalikan hasil vektorisasi
