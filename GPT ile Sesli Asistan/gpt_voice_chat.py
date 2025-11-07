"""
Problem tanimi: Gpt ile sesli sohbet
    - kullanicinin mikrofona konusarak soru sormasi
    - openai whisper modeli
    - metnin gpt 4.1 nano ile analiz edilmesi
    - guvenlik: zararli dil filtreleme

Kullanilan Teknolojiler:
    - ses kaydi
    - ses -> metin: openai whisper modeli
    - cevap uretimi: openai gpt 4.1 nano
    - loglama: loglama modulu
    - zararli icerik fitreleme: kufurler

Model tanimi: Openai whisper, GPT 4.1 nano
    - Cok dilli konusma tanima modeli
    - Konusmalari yaziya doker
    - Birden cok dili destekler
    - otomotik dil algilama yapabilir
    - whisper lite:
        offline ve acik kaynakli

api tanimi:

plan/program:

install libraries:

import libraries:
"""
#import libs
from openai import OpenAI
import sounddevice as sd #mikrofon erisimi
from scipy.io.wavfile import write # ses kaydini wav dosyasina yazma araci
import os #dosya islemleri
import uuid #benzersiz kimlik
import re # zararli icerik filtreleme
from datetime import datetime #tarih ve saat
from dotenv import load_dotenv #.env yukleme
import logging

#log
now = datetime.now().strftime("%Y%m%d_%H%M%S") #dosya adi icin suaki zaman
log_file = f"logs/konusma_{now}.log" #log dosyasi adi

#log klosoru yoksa olustur
os.makedirs("logs", exist_ok=True)

#log formati ve seviyesi: DEBUG, INFO, WARNING, ERROR, CRITICAL
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [
        logging.FileHandler(log_file, encoding='utf-8'), #log yazdir
        logging.StreamHandler() #konsola yazdir
    ]
)
logger = logging.getLogger(__name__)

#.env
load_dotenv() #.env dosyasini yukle
client = OpenAI() #openai istemcisi

DURATION = 5  # saniye cinsinden kayit suresi
FS = 44100  # ornekleme hizi

#zararli kelime filtre
BANNED_WORDS = ["mal", "salak", "gerizekalı"]  # ornek zararli kelimeler

def filter_bad_words(text):
    filtered_text = text
    for word in BANNED_WORDS:
        if re.search(rf"\b{word}\b", text, flags = re.IGNORECASE): 
            logger.warning(f"Zararlı kelime tespit edildi: {word}") 
        filtered_text = re.sub(rf"\b{word}\b", "***" * len(word), filtered_text, flags = re.IGNORECASE) 

    return filtered_text

# filter_bad_words("Bu adam tam bir gerizekalı")  2025-11-07 13:28:27,642 [WARNING] Zararlı kelime tespit edildi: mal

#ses kaydi alma ve whisper ayarlari, sesi metne cevirme ve kaydetme
def record_audio(filename = "recorded.wav", duration = DURATION):
    logger.info("Ses kaydı başlatılıyor...")
    recording = sd.rec(int(duration * FS), samplerate = FS, channels = 1)
    sd.wait()
    write(filename, FS, recording)
    logger.info(f"Ses kaydı tamamlandı: {filename}")

def transcribe_with_whisper(audio_path):
    logger.info("Whisper ile ses metne dönüştürülüyor...")
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file = audio_file,
            model = "whisper-1",
            response_format = "text",
            language = "tr"  
        )
    return transcript

def get_gpt_response(messages):
    logger.info("GPT 4.1 nano ile cevap üretiliyor...")
    response = client.chat.completions.create(
        model = "gpt-4.1-nano",
        messages = messages
    )
    return response.choices[0].message.content # ilk cevabi al

# print(f"GPT YANIT TEST: {get_gpt_response([{'role': 'user', 'content': 'Merhaba, nasılsın?'}])}") 
#2025-11-07 13:43:24,522 [INFO] GPT 4.1 nano ile cevap üretiliyor...
#2025-11-07 13:43:29,053 [INFO] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
#GPT YANIT TEST: Merhaba! İyiyim, teşekkür ederim. Siz nasılsınız? Size nasıl yardımcı olabilirim?

#hepsini birlestir

if __name__ == "__main__":
    logger.info("Sesli Asistan Başlatıldı.")
    logger.info(f"Konusma log Dosyası: {log_file}")

    # mesaj gecmisini sistem mesajiyla baslat
    messages = [
        {"role": "system", "content": "Sen yardımcı bir sesli asistansın. Kullanıcıya kibar ve yardımcı cevaplar ver."}
    ]

    #while True: #kullanici cikana kadar devam et
    uid = str(uuid.uuid4())[:8]  # benzersiz kimlik, ilk 8 karakter
    audio_file = f"record_{uid}.wav" # gecici wav dosyasi

    record_audio(audio_file, DURATION) #ses kaydi al
    question = transcribe_with_whisper(audio_file) #whisper ile metne cevir
    logger.info(f"Kullanıcı ({uid}) Soru: {question}")

    filtered_question = filter_bad_words(question) #zararli icerik filtrele
    if filtered_question != question: #filtreleme yapildiysa
        logger.info(f"Kullanıcı sorusu zararlı içerik nedeniyle filtrelendi: {filtered_question}")

    if "çık" in filtered_question.lower(): #kullanici cikmak isterse
        logger.info("Kullanıcı çıkış yaptı. Sesli Asistan sonlandırılıyor.")

    messages.append({"role": "user", "content": filtered_question}) #kullanici mesajini ekle
    answer = get_gpt_response(messages) #gpt cevabini al
    
    logger.info(f"GPT Cevap: {answer}")

    os.remove(audio_file) #gecici dosyayi sil

    logger.info("Sesli Asistan Sonlandırıldı.")