# Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Veri yükleme
fake_news = pd.read_csv('fake_news.csv')
true_news = pd.read_csv('true_news.csv')

# Sahte ile gerçek haberler için 1 ve 0 etiketlerinin girilmesi
fake_news['label'] = 0
true_news['label'] = 1

# Sahte ve gerçek haberlerin birleştirilmesi
news = pd.concat([fake_news,true_news], axis=0)

# Boş veri olup olmadığının kontrolü
#print(news.isnull().sum())

# Title ile text özniteliklerinin birleştirilmesi
news['text'] = news['title'] + ' ' + news['text']

# Gereksiz tarih özniteliğinin ve boş olan eski title'ın silinmesi
news = news.drop(['date', 'title'], axis=1)

# Sahte ve gerçek haberlerin sırayla gelmemesi için karıştırılması
news = news.sample(frac=1)

# Karıştırıldıktan sonra oluşan index sayılarını düzenleme
news.reset_index(inplace=True)
news.drop(['index'], axis=1, inplace=True)

# Verilerin Ön İşlemesi
def word_operations(text):
    # Küçük harfe dönüştürülmesi
    text = text.lower()
    
    # Linklerin silinmesi
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # HTML Taglerinin silinmesi
    text = re.sub(r'<.*?>', ' ', text)
    
    # Noktalama işaretlerinin silinmesi
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Sayıların silinmesi
    text = re.sub(r'\d', ' ', text)
    
    # Yeni satır karakterlerinin silinmesi (\n)
    text = re.sub(r'\n', ' ', text)
    
    # Sekme karakterlerinin silinmesi (\t)
    text = re.sub(r'\t', ' ', text)
    
    # Fazla boşlukların silinmesi
    text = re.sub(r'  ', ' ', text)
    
    # Özel karakterlerin silinmesi
    text = re.sub(r'[!@#$%^&*()_+-={}[]|:;"\'<>,.?/~\]', ' ', text)
    
    return text

# Etkisiz kelimeleri kaldırma ve Lemmatization işlemleri için gerekli kütüphaneler
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print(stop_words)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# POS tag'i WordNet formatına dönüştüren yardımcı fonksiyon
def get_wordnet_pos(tag):
    if tag.startswith('J'):  # adjective(sıfat)
        return wordnet.ADJ
    elif tag.startswith('V'):   # verb(fiil)
        return wordnet.VERB
    elif tag.startswith('N'):   # noun(isim)
        return wordnet.NOUN
    elif tag.startswith('R'):   # adverb(zarf)
        return wordnet.ADV
    else:
        return None

# Etkisiz kelimelerin silinmesi
def remove_stop_words(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

# Stemming / Lemmatization (kelime köküne inilmesi)
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    # Metni kelimelere ayırma
    words = word_tokenize(text)
    # Kelimelere POS tag ekleme
    words_with_pos = nltk.pos_tag(words)
    # Her bir kelime için lemmatization uygulama
    lemmatized_text = []
    for word, pos in words_with_pos:
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN  # Default olarak NOUN al
        lemmatized_text.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return ' '.join(lemmatized_text)