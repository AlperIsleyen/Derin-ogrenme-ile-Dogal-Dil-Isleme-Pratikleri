"""
problem tanimi: Lstm ile metin uretme

        - ben yarin ....... (Ne gelmeli)

lstm: long short term memory

veri seti: chatgpt ile olusturulmus 100 adet gunluk hayat cumlesi

install libraries pip , requirement.txt

import libraries

"""
#import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# chatgpt ile egitim verisi

data = [
    "Bugün hava top oynamak için çok güzel",
    "Ben dün gidemedim çünkü çok uyudum kaldım",
    "Sen dün sinemaya gittin mi gitmedin mi bilmiyorum",
    "Ben hiç o filmi izlemedim daha önce",
    "Karnım çok açtı ama yemek yapmadım dün",
    "O kadar güzel bir gündü ki dışarı çıkmadık",
    "Telefonum şarjı yine bitti gitti",
    "Okula erken gittim ama öğretmen gelmedi daha",
    "Ben kitap okuyorumdu o sırada",
    "Yarın gelirim belki de gelmem bilmiyorum",
    "Çok susadım ama su kalmadı evde hiç",
    "Sen dün ders çalıştın mıydı",
    "Arabayı park ettim nereye ettiğimi unuttum",
    "O kadar yorgunum ki gözlerim açık duramıyor",
    "Ben seninle konuşmak istiyor gibi hissediyorum",
    "Film çok sıkıcıydı ama sonu güzeldi gibi",
    "Arkadaşım geldi ama erken gitti çok erken",
    "Benim karnım tok ama yine de yemek istiyorum",
    "Sen bana hiç mesaj atmıyorsun artık neden",
    "Annem dedi ki dışarı çıkma ama çıktım",
    "Bugün market gitmeyi unutum",
    "Kedim bugün hiç yemedim gibi davranıyor",
    "Okuldan geldikten sonra hemen uyudum kaldım",
    "Ben dün seninle konuştu muydum",
    "Telefonum çaldı ama duymadım duydum sanmışım",
    "Sana dün demedim mi dikkat et diye",
    "Ders çok sıkıcıydı nerdeyse uyuyordum derste",
    "Bugün sabah geç kalktım sonra yine uyudum",
    "Ben onu dün görmüştüm ama o beni görmemiş",
    "Yemek çok tuzlu olmuş biraz daha tuz koydum",
    "Kardeşim bana bağırdı ama sonra özür dilemedi hiç",
    "Yarın erken kalkmam lazım ama kalkmam büyük ihtimalle",
    "Bu hafta sonu dışarı çıkacağım eğer yağmur olmazsa belki",
    "Dün çok güzel bir rüya gördüm ama hatırlamıyorum şimdi",
    "Ben daha kahvaltı bile yapmadım ya",
    "Ders çalışacağım diyordum ama diziye başladım",
    "Bilgisayar yine dondu açılmıyor hiçbir şey",
    "Sana demedim mi gitme oraya diye",
    "Bugün çok güzel geçti ama sanki bir şey eksikti",
    "Çantamı evde unuttum sonra geri döndüm almak için",
    "O kadar uykum var ki ama uyuyamıyorum",
    "Ben seni bekledim ama sen gelmedin hiç",
    "Arkadaşım aradı ama telefonu kapandı hemen",
    "Bu hafta hava çok sıcak olacak diyorlar ama inanmam",
    "Bugün dışarı çıkacaktık ama herkes iptal etti",
    "Kahvemi döktüm sonra yenisini yaptım sonra yine döktüm",
    "Yemek çok güzel olmuştu ama ben tokumdu",
    "Senin söylediklerini anlamadım tekrar söylesene",
    "O kadar çok konuştuk ki sesim gitti artık",
    "Ben dün yemeğe gitmedi gitmedim sanırım",
    "Sabah kahvaltı yapmadan çıktım sonra çok acıktım",
    "Bugün hava bir garip sıcak gibi değil gibi",
    "Ben senin mesajını gördüm ama cevap yazmadım unuttum",
    "Kafam çok dolu hiçbir şeye odaklanamıyor gibiyim",
    "Yemek pişmedi daha ama kokusu geliyor gibi",
    "Dün gece film izliyordum birden elektrik gitti geldi",
    "Bu hafta çok yoruldum artık dinlenmek istiyorum sadece",
    "Kedim bütün gece miyavladı uyuyamadım hiç",
    "Arkadaşım bana dedi ki sen çok değiştin diyor",
    "Sana dün söyledim zaten ama unuttun galiba yine",
    "Markete gittim ama ekmek kalmamış hiç kalmamış",
    "Ben kitap okumayı seviyorum ama bazen sıkılıyorum hemen",
    "O kadar yorgunum ki yataktan kalkasım yok",
    "Bu sabah kahvemi döktüm sonra yeni yaptım tekrar döktüm",
    "Yarın toplantı var sanıyordum ama değilmişmiş meğer",
    "O kadar acele ettim ki yine geç kaldım",
    "Ben seni dün aradım ama açmadın hiç",
    "O kadar açtım ki iki tabak yedim yine doymadım",
    "Bugün işe gitmek istemedim ama gittim mecburen",
    "Bilgisayarım çok yavaşladı artık dayanamıyorum buna",
    "Sabah kalktım kahve koydum sonra tekrar uyudum",
    "Sana dün geleceğim dedim ama işler çıktı kusura bakma",
    "Telefonumun ekranı kırıldı ama çalışıyor yine de biraz",
    "Benim cüzdan yine evde unuttum galiba",
    "Yemek yapacaktım ama malzeme kalmamış evde",
    "Kardeşim odaya girdi ışığı kapattı sinir oldum",
    "Bütün gün çalıştım ama hiçbir şey bitmedi gibi",
    "Yağmur yağacak sandım ama hiç damla düşmedi",
    "Ben dizi izliyordum ama sonunu göremedim uyuyakaldım",
    "Bugün çok güzel geçti ama akşam canım sıkıldı birden",
    "Yarın erken kalkmam gerek ama uykum hiç yok",
    "O kadar üşüyorum ki battaniye bile yetmiyor artık",
    "Sen bana fotoğraf atmadın mıydı dün akşam",
    "Yemek yedikten sonra tatlı canım istedi çok fena",
    "Kedim koltuğu yine tırmalamış mahvetmiş",
    "Bugün kimseyle konuşmak istemiyorum hiç",
    "Telefon çaldı ama numarayı tanımadım açmadım",
    "O kadar sıkıldım ki dışarı çıkasım bile yok",
    "Marketten su alacaktım ama param kalmamış cidden",
    "Ben seni gördüm sandım ama başkasıymış meğer",
    "Bugün çok rüzgar var saçım mahvoldu resmen",
    "O kadar geç kaldım ki otobüsü kaçırdım yine",
    "Bilgisayar güncelleme yaptı ama kapanmadı saatlerce",
    "Arkadaşım beni çağırdı gittim ama kimse yoktu orada",
    "Yarın yağmur yağacakmış diyorlar ama sanmam bence",
    "Bugün erken yattım ama geç uyudum yine",
    "Kahve yapacaktım ama su ısıtıcı çalışmadı ki",
    "Benim çantamı gördün mü hiçbir yerde bulamıyorum",
    "Sabah işe geç kaldım çünkü alarm çalmadı",
    "Bugün çok güzel bir gündü ama çok da yorucuydu"
]


# -- Preprocessing --
# kelimeleri indexe cevir (tokenize)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1 # +1 padding

print(f"total_words: {total_words}")

#n-gram dizileri olustur her cumleden kisa dizi olustur (embedding)
input_sequences = []
for text in data:
    token_list = tokenizer.texts_to_sequences([text])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i+1]
        input_sequences.append(n_gram_sequence)


#padding farkli uzunluklari sabitle
max_sequence_length = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen = max_sequence_length, padding ="pre")

print(f"after padding sequences: \n{input_sequences}")

"""
[  0   0   0 ...   0   5  38]
 [  0   0   0 ...   5  38 106]
 [  0   0   0 ...  38 106 107] ...    #bosluklar 0 oldu

"""

# girdi (X) ve hedef degiskenler (y) ayir
X = input_sequences[:, :-1] # n - 1 kelimeyi giris olarak sec
y = input_sequences[:, -1] # n inci kelimeyi tahmin et


# hedef degiskene one hot encoding
y = tf.keras.utils.to_categorical(y, num_classes = total_words)
print(f"hedef degisken: {y}")

"""
hedef degisken: [[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 1. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 1. 0.]
 [0. 0. 0. ... 0. 0. 1.]]

"""
# lstm model tanimla
model = Sequential()
model.add(Embedding(total_words, 50, input_length = X.shape[1]))
model.add(LSTM(100))
model.add(Dense(total_words, activation = "softmax")) #cok sinifli her kelime kadar sinif "354"

"""
X = [bugün hava cok]
y = [güzel]
"""

# compile
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
print(model.summary())


# egitim training
model.fit(X, y, epochs = 100, verbose = 1)  #verbose: egitim surecini konsolda yazdirma
"""
Epoch 97/100
19/19 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.8750 - loss: 0.5172 
Epoch 98/100
19/19 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.8717 - loss: 0.5092 
Epoch 99/100
19/19 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.8717 - loss: 0.5007 
Epoch 100/100
19/19 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.8734 - loss: 0.4933 

Model ezber yapti 100 kelimenin 87sini dogru yapti
"""

# ornek uret

def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list= tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen = max_sequence_length -1, padding= "pre") # padding bosluklar 0 oldu
        predicted_probs = model.predict(token_list, verbose = 0)
        predicted_index = np.argmax(predicted_probs, axis = -1)[0]
        predicted_word = tokenizer.index_word[predicted_index] #sayilarda worde
        seed_text = seed_text + " " + predicted_word #bir sonrakini tahmin et ve seed_texti guncelle
    return seed_text
    
print(generate_text("Bugün hava", 5))

"""
(1)
seed_text = bu sabah
predicted_word = okula

(2)
seed_text = bu sabah okula
predicted_word = geç

(return)
seed_text = bu sabah okula geç


Bugün hava
Bugün hava top oynamak için çok güzel
"""