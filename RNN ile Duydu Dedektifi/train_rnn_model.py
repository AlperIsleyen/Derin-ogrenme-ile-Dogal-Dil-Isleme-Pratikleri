"""
Problem Tanimi: Yorum olumlu mu, olumsuz mu anlamak.

    this movie is awesome --> pozitif
    it was terrible movie --> negatif

RNN: Tekrarlayan sinir aglari, onceki veriden hatirlayarak cevap verir

Girdi: film -> cok -> kotuydu
Bellek:
Cikti: anlam anlam olumsuz

Veri Seti: IMDB veri seti: film yorumlari (Olumlu ve olumsuz)
    - 50000 adet film yorumu,
    - 0 negatif, 1 pozitif,
    - great = 65


plan/program:



Gerekli Kurulumlar:


Import Libraries

"""

#import libraries
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential # base model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense 
from tensorflow.keras.datasets import imdb # veri seti
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Stopwords listesi belirle
nltk.download("stopwords") #ingilizce stopwords
stop_words = set(stopwords.words("english")) # stopwords set edildi

# Model parametreleri

max_features = 10000 # en cok kullanilan 10 bin kelime
maxlen = 500

# load dataset

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = max_features) # train test ayrilmis olarak gelir


#ornek veri incelemesi

original_word_index = imdb.get_word_index()

#Tokenize
inv_word_index = {index + 3: word for word, index in original_word_index.items()}
inv_word_index[0] = "<PAD>" # 0: bosluk:padding
inv_word_index[1] = "<START>" # 1: cümle baslangici
inv_word_index[2] = "<UNK>" # 2: bilinmeyen kelime

#sayi dizilerilerini kelimeleri cevir

def decode_review(encoded_review):
    return " ".join([inv_word_index.get(i, "?") for i in encoded_review])

movie_index = 0

#ilk eğitim verisini yazdır

print("ilk yorum: (sayi dizisi)")
print(X_train[movie_index])

print("ilk yorum: (kelimelerle)")
print(decode_review(X_train[movie_index]))

print(f"Label: {"Pozitif" if y_train[movie_index]== 1 else "Negatif"}")

#Gerekli sozcukleri oluştur: word to index ve index to word

word_index = imdb.get_word_index()
index_to_word = {index + 3: word for word, index in word_index.items()} # sayilardan kelimelere gecis
index_to_word[0] = "<PAD>" # 0: bosluk:padding
index_to_word[1] = "<START>" # 1: cümle baslangici
index_to_word[2] = "<UNK>" # 2: bilinmeyen kelime
word_to_index = {word: index for index, word in index_to_word.items()} #kelimelerden sayilara

#onisleme veri temizleme (data preprocessing)

def preprocess_review(encoded_review):
    #sayilari kelimelere cevir
    words = [index_to_word.get(i,"") for i in encoded_review if i >= 3]

    #sadece harflerden oluşan ve stopword olmayanlari al
    cleaned = [
        word.lower() for word in words if word.isalpha() and word.lower() not in stop_words
    ]

    #tekrardan temizlenmisleri sayilara cevir
    return [word_to_index.get(word, 2) for word in cleaned]

#veri temizleme ve padding

X_train = [preprocess_review(review) for review in X_train]
X_test = [preprocess_review(review) for review in X_test]

#pad bolumu
"""
merhaba bugun hava cok guzel
merhaba, naber, 0, 0, 0          padding yani sonlari yapti hepsini esitledi
"""

X_train = pad_sequences(X_train, maxlen = maxlen)
X_test = pad_sequences(X_test, maxlen = maxlen)

#RNN modeli olusturma

model = Sequential() #base model: katmanlari sirali olarak ekle

#embedding 32 boyutlu vektore donustur

model.add(Embedding(input_dim = max_features, output_dim = 32, input_length = maxlen))

#simplernn katmani: metni sirala isler ve baglam iliskisini ogrenir

model.add(SimpleRNN(units = 32)) #cell noron sayisi overfitting olursa azalt

#output katmani: siniflandirma binary classification bu yüzden sigmoid(0 ve 1 arasi deger) kullaniyoruz

model.add(Dense(1, activation = "sigmoid"))

# model compile
model.compile(
    optimizer = "adam", #agirlik guncellemesi
    loss = "binary_crossentropy", #kayip fonksiyonu
    metrics = ["accuracy"] #degerlendirme
)

print(model.summary())

#Egitim, training

history = model.fit(
    X_train, y_train, #girdi ve cikti
    epochs = 2, # tekrar
    batch_size = 64, # 64lu paketler halinde ogren
    validation_split = 0.2 # %20 dogrulama icin ayir
)

#Model evaluation

def plot_history(hist):
    plt.figure(figsize = (12,4))

    #accuracy
    plt.subplot(1,2,1)
    plt.plot(hist.history["accuracy"], label = "Training")
    plt.plot(hist.history["val_accuracy"], label = "Validation")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    #loss plot
    plt.subplot(1,2,2)
    plt.plot(hist.history["loss"], label = "Training")
    plt.plot(hist.history["val_loss"], label = "Validation")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

test_loss, test_acc = model.evaluate(X_test,y_test) #test
print(f"Test: {test_acc:.2f}")

#kaydet
model.save("rnn_duygu_model.h5")
print(f"Model basariyla kaydedildi: rnn_duygu_model.h5")