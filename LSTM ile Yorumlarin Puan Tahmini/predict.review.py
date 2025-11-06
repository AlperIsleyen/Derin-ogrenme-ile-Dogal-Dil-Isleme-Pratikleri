"""
yorumu tahmin etsin

"ben bir doktora gittim ve bu doktoru çok sevdim." Puan 4.5 gibi
"""

import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Model
model = load_model("regression_lstm_yelp.h5", compile = False) # Compile: tekrar train etmeyecegim

#Tokenizer tanimla
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

#ornek input ilki 1 digeri 5 yildiz
texts = [ 
    "Terrible. Preordered my tires and when I arrived they couldn't find the order anywhere. Once we got through that process I waited over 2 hours for them to be put on... I was originally told it would take 30 mins. Slow, over priced, I'll go elsewhere next time.",
    "Cheap, unpretentious, and, for this, one of my favorite breakfast joints in the country. Simply put I LOVE it here. The mixed grill, the sausage and egg on a biscuit, the home fries. This it the very definition of diner. Thank you Gab 'n' Eat!"
]

# Tokenizer, padding
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen = 100, padding = "post")

#LSTM
predictions = model.predict(padded)

#Post Processing
predictions_scaled = predictions * 5

#Sonucu yazdir
for i, comment in enumerate(texts):
    print(f"Yorum: \n{comment}")
    print(f"Tahmini skor degeri: {predictions_scaled[i][0]:.2f}")

"""
Yorum:
Terrible. Preordered my tires and when I arrived they couldn't find the order anywhere. Once we got through that process I waited over 2 hours for them to be pu
t on... I was originally told it would take 30 mins. Slow, over priced, I'll go elsewhere next time.
Tahmini skor degeri: 0.10
Yorum:
Cheap, unpretentious, and, for this, one of my favorite breakfast joints in the country. Simply put I LOVE it here. The mixed grill, the sausage and egg on a biscuit, the home fries. This it the very definition of diner. Thank you Gab 'n' Eat!
Tahmini skor degeri: 4.33
"""