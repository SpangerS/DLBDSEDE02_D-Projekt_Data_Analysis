# Import notwendiger Bibliotheken und Methoden
import spacy
import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

# Laden des deutschen SpaCy-Modells, für den Lemmatizer
nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])

# Definition für die später genutzte Funktion zur Lemmatisierung, Kleinschreibung und Entfernung von Stoppwörtern
def german_lemmatizer(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc 
            if not token.is_stop
            and not token.is_punct 
            and not token.is_digit
            and not token.is_space
            and not token.like_url
            and not token.like_email
            and not token.like_num]

# Datenkorpus aus CSV importieren & Zählvariable für Anzahl Artikel definieren & mitzählen
corpus = []
a1 = 0
with open("C:/Temp/testdata.csv", encoding='utf-8', errors='replace') as csvdatei:
    csv_reader_object = csv.reader(csvdatei)
    for row in csv_reader_object:
        corpus.append(', '.join(row))
        a1 += 1

# LSA & LDA basieren auf der folgenden Tf-idf-Vektorisierung, sowie der Anzahl Komponenten/Themen bei variablen Input-CSVs, daher wird a2 eingeführt 
tfidf_vect = TfidfVectorizer(analyzer=german_lemmatizer, min_df=0.05, use_idf=True, max_features=a1, smooth_idf=True)
tfidf_model = tfidf_vect.fit_transform(corpus)
a2 = 0

# Anzahl für n_components festlegen - sollte maximal ein hundertstel von (Anzahl der Artikel) sein
a2 = int(a1 / 100)

# LSA-Modell erstellen
lsa_model = TruncatedSVD(n_components=a2, algorithm='randomized', n_iter=1)
lsa = lsa_model.fit_transform(tfidf_model)

# LDA-Modell erstellen
lda_model = LatentDirichletAllocation(n_components=a2, learning_method='online', random_state=42, max_iter=1)
lda_top=lda_model.fit_transform(tfidf_model)

print('Für das LDA-Modell wählen wir die probabilistische Methode durch Summieren der Themenverteilung im Datensatz, so werden multiple Themen je Artikel berücksichtigt.')
print('')

# Darstellung der häufigsten Themen mit den jeweiligen Top-10 Worten
print('Darstellung der häufigsten Themen mit den jeweiligen Top-10 Worten:')
print('')
w = tfidf_vect.get_feature_names_out()
for top_idx, top in enumerate(lda_model.components_):
    print(f"Thema #{top_idx}:")
    print(" ".join([w[i] for i in top.argsort()[:-10 - 1:-1]]))
print('')

# Transformieren der Themenmatrix, Summieren der Themen, Sortieren nach Häufigkeit und Ausgeben der Daten
art_top = lda_model.transform(tfidf_model)
top_sum = art_top.sum(axis=0)
sort_top_i = np.argsort(top_sum)[::-1]

print("Häufigste Themen (Index):", sort_top_i)
print("Summe der Wahrscheinlichkeiten:", top_sum[sort_top_i])
input("Drücken Sie Enter, um das Fenster zu schließen oder das Programm zu beenden.")
