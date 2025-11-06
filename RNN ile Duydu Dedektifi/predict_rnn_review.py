"""
Egitilmis model kullanarak analiz edelim
"""

import numpy as np
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

#model parametreleri

max_features = 10000 #kullanilan max kelime sayisi
maxlen = 500 #modelin max uzunlugu

#stopwordsden kurtul
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

#imdb verisetinden kelimeler aliniyor
word_index = imdb.get_word_index()

#sayi -> kelime
index_to_word = {index + 3: word for word, index in word_index.items()} # sayilardan kelimelere gecis
index_to_word[0] = "<PAD>" # 0: bosluk:padding
index_to_word[1] = "<START>" # 1: cümle baslangici
index_to_word[2] = "<UNK>" # 2: bilinmeyen kelime
word_to_index = {word: index for index, word in index_to_word.items()} #kelimelerden sayilara

#modeli yukle
model = load_model("rnn_duygu_model.h5")
print("Model basariyla yuklendi")

#tahmin yap
def predict_review(text):
    """
    kullanicidan gelen metni temizle ve modele uygun hale getir sonra tahmin et
    """

    #hepsini kucuk harf yap
    words = text_to_word_sequence(text)

    #stopwords cikarma ve sadece kelime alma

    cleaned = [
        word.lower() for word in words if word.isalpha() and word.lower() not in stop_words
    ]

    # her kelime egitilen sozlukten sayiya cevrilir
    encoded = [word_to_index.get(word, 2) for word in cleaned] # 2 = <UNK>

    #Modelin bekledigi sabit uzunluk
    padded = pad_sequences([encoded], maxlen = maxlen)

    # Tahmin
    prediction = model.predict(padded)[0][0]

    print(f"Pozitif tahmin olasiligi: {prediction:.4f}")
    if(prediction > 0.5):
        print("Pozitif")
    else: print("Negatif")

#cumle gir ve predict et
user_review = input("Bir film yorumu girin")
predict_review(user_review)
