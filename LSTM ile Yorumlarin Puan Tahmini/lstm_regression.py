"""
Problem Tanimi -> puan tahmini (1-5), regresyon problemi
    - cok iyiydi cok memnun kaldim -> 4.5
    - berbatti, bir daha gelmem -> 1.2

Veri Seti: yelp dataset, hugging face, (doktor, restoran, otel, araba yikama...)
    - text: yorum metni
    - label: 0-4 arasinda ama bunu 1 ile 5e cekelim

LSTM: Bir yorumu bastan sona okur ve puanlar

install libraries, requirements.txt: pandas, numpy, matplotlib, scikit-learn, fsspec, huggingface-learn

import libraries


"""
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle # tokenizeri diske kaydetmek amacli

from sklearn.model_selection import train_test_split # veriyi egitim ve test olmak uzere ikiye ayirmak icin
from sklearn.preprocessing import MinMaxScaler # Normalizasyon

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import MeanSquaredError   # Regresyonda 
from tensorflow.keras.metrics import MeanAbsoluteError   # Regresyonda hata orani hesabi icin 5 -> 4 = 1, 4 -> 2 = 2 --> (1+2)/2

#load yelp dataset
splits = {'train': 'yelp_review_full/train-00000-of-00001.parquet', 'test': 'yelp_review_full/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/Yelp/yelp_review_full/" + splits["train"])
print(df.head())

#etiketleri 0-4 araliginda 1-5 araligina donustur
df["label"] = df["label"] + 1

#data preprocessing
texts = df["text"].values  # yorum metinleri
labels = df["label"].values  # puanlar 1-5

# tokenizer
# num_words en çok geçen 10000 kelime
# oov = bilinmeyen kelimeleri etiketleme
tokenizer = Tokenizer(num_words = 10000, oov_token = "<OOV>")

#metni sayilara donustur
tokenizer.fit_on_texts(texts)

#tokenizeri diske kaydet
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

#yorumlari dizi haline getir
sequences = tokenizer.texts_to_sequences(texts)

#padding
padded_sequences = pad_sequences(sequences, maxlen = 100, padding = "post", truncating = "post")

#normalizasyon
scaler = MinMaxScaler() # 1-5 sonra -1 ve /4 --> 0-4
labels_scaled = scaler.fit_transform(labels.reshape(-1, 1))

#egitim ve test
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels_scaled, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_train: {X_train[:2]}")
print(f"y_train shape: {y_train.shape}")
print(f"y_train: {y_train[:2]}")

"""
X_train shape: (520000, 100)
X_train: [[   5  131  274  148  212   10   14    9   13    4  128   86   41   12
   959   46    8    2  108  271    5 1038   50    5  824    6 1408    5
   617  363   38    8    2  729    3 1015    4  405    8  619   19  122
   979  476   11    2  128    3 1084 2985   38   63   13    2  414   11
  5550   19  175  469    6  464   69 1267    6    2  128  174    5   24
     4 1267    3   17 2496  132    9   38   10    4 3378    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0]
 [  30   13   51    5  214  144   96   10    2   54    3    2  787  127
   853 2863 3575   73   19    4   63    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0]]
y_train shape: (520000, 1)
y_train: [[0.5]
 [0. ]]

 0'lar padding
"""

#LSTM tabanli regresyon modeli
model = Sequential()
#input => kelime sayısı; lenght => sabit dizi uzunluğu
#output => her kelime 128 boyutlu olacak
model.add(Embedding(input_dim=10000, output_dim = 128, input_length = 100))

#LSTM katmanı
model.add(LSTM(128)) #128 hucre daha fazla ogrenme kapasitesi

#fully connected layer (dense)
model.add(Dense(64, activation = "relu")) 

#output layer
model.add(Dense(1, activation = "linear"))  #relu, tanh, sigmoid (2 sınıflı), softmax, linear

# model compile and training
model.compile(
    optimizer = "adam", #adaptif ogrenme algoritmasi
    loss = MeanSquaredError(), #genelde regrestonda kullanilir, yorumlamayi kolaylastirir
    metrics = [MeanAbsoluteError()] # hata ortalamasi
)

history = model.fit(
    X_train, y_train,
    epochs = 3,
    batch_size = 64,
    validation_split = 0.2 # %20 validasyona gidiyor
)

# egitim loss graph and save model
plt.plot(history.history["loss"], label = "Training Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.title("Egitim sureci MSE")
plt.xlabel("Epoch")
plt.ylabel("Loss MSE")
plt.show()

model.save("regression_lstm_yelp.h5")