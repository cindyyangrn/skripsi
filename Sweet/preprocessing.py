import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import string
import re

class Preprocessor:
    def __init__(self):
        # Inisialisasi dictionary normalisasi
        self.normalization_dict = {
            "sih": "", "dong": "", "kah": "", "kok": "", "dah": "", "deh": "", "nya": "", "loh": "",
            "ni": "", "nih": "", "tuh": "", "min": "", "kan": "", "ya": "", "xixi": "", "xixixi": "",
            "si": "", "donk": "", "kl": "kalau", "klo": "kalau", "kalo": "kalau", "bgt": "sekali",
            "yg": "yang", "yng": "yang", "bat": "sekali", "bett": "sekali", "bet": "sekali", "begete": "sekali",
            "gk": "tidak", "gak": "tidak", "ga": "tidak", "nggak": "tidak", "nggk": "tidak", "tdk": "tidak",
            "aja": "saja", "ttp": "tetap", "masi": "masih", "liat": "lihat",
            "tp": "tapi", "tpi": "tapi", "dr": "dari", "dri": "dari",
            "dapet": "dapat", "dpt": "dapat", "gmn": "bagaimana", "gmna": "bagaimana",
            "gimana": "bagaimana", "blm": "belum", "blom": "belum", "belom": "belum",
            "jga": "juga", "jg": "juga", "jngn": "jangan", "lbh": "lebih", "blh": "boleh",
            "w": "aku", "gw": "aku", "gua": "aku", "gwh": "aku", "guweh": "aku",
            "knp": "kenapa", "kpn": "kapan", "dgn": "dengan", "bgt": "banget",
            "pdhl": "padahal", "udah": "sudah", "dlu": "dulu", "tau": "tahu",
            "msh": "masih", "krn": "karena", "karna": "karena", "seneng": "senang",
            "buat": "untuk", "buatt": "untuk", "banget": "sangat", "bangett": "sangat",
            "sm": "sama", "btl": "betul", "mcm": "macam", "org": "orang",
            "bsk": "besok", "udh": "sudah", "inves": "investasi", "invest": "investasi",
            "gajelas": "tidak jelas", "gaada": "tidak ada", "gada": "tidak ada",
            "skrg": "sekarang", "sampe": "sampai", "gausa": "tidak usah",
            "gabisa": "tidak bisa", "napa": "kenapa", "gitu": "begitu"
        }
        # Inisialisasi objek untuk stopwords removal
        stopword_factory = StopWordRemoverFactory()
        self.stopwords = stopword_factory.get_stop_words()

    def clean_text(self, text):
        """
        Membersihkan teks dari tanda baca, normalisasi, dan stopwords.
        """
        if not isinstance(text, str):
            return ""
        # Mengubah teks menjadi huruf kecil
        text = text.lower()
        # Menghapus tanda baca
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Menghapus simbol-simbol acak yang bukan huruf atau angka
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Tokenisasi teks
        tokens = word_tokenize(text)
        # Melakukan normalisasi
        tokens = [self.normalization_dict.get(word, word) for word in tokens]
        # Menghapus stopwords
        tokens = [word for word in tokens if word not in self.stopwords]
        # Menggabungkan kembali token menjadi string
        cleaned_text = ' '.join(tokens)
        return cleaned_text

    def clean_dataframe(self, df, text_column):
        """
        Membersihkan data pada kolom tertentu dalam DataFrame.
        """
        # Menghapus baris yang kosong
        df = df.dropna(how='all')
        # Menghapus kolom yang kosong
        df = df.dropna(axis=1, how='all')
        # Terapkan clean_text pada kolom yang diinginkan
        df[text_column] = df[text_column].apply(self.clean_text)
        # Hapus baris dengan teks kosong setelah pembersihan
        df = df.dropna(subset=[text_column])
        return df
