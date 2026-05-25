# Import notwendiger Bibliotheken und Methoden
import spacy
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

# Laden des deutschen SpaCy-Modells, für den Lemmatizer
nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])

# Abfrage des indivduellen Datenpfades für die Quell-CSV
pfad = input("Bitte geben Sie den Quelldatensatz (CSV-File) mit vollständigem Pfad an: ")
print("Der angegebene Pfad lautet: " + pfad)

# Abfrage eines individuellen Output-Pfades für die Vektoren und Definition der Files
outputpfad = input("Bitte geben Sie einen Pfad - inklusive Slash (Linux) oder Backslash (Windows) - für den CSV-Output der erzeugten Vektoren an: ")
print("Der angegebene Output-Pfad lautet: " + outputpfad)
bow_output = outputpfad + "BoW_vektoren.csv"
tfidf_output = outputpfad + "Tfidf_vektoren.csv"
print("Folgende Dateien werden durch dieses Skript erstellt: ")
print(bow_output)
print(tfidf_output)

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
with open(pfad, encoding='utf-8', errors='replace') as csvdatei:
    csv_reader_object = csv.reader(csvdatei)
    for row in csv_reader_object:
        corpus.append(', '.join(row))
        a1 += 1

# Daten als BoW-Vektor vektorisieren und in Array speichern
bow_vect = CountVectorizer(analyzer=german_lemmatizer, min_df=0.05)
bow_model = bow_vect.fit_transform(corpus)
bow_data=pd.DataFrame(bow_model.toarray(), columns=bow_vect.get_feature_names_out())

# Daten als Tf-idf-Vektor vektorisieren und in Array speichern
tfidf_vect = TfidfVectorizer(analyzer=german_lemmatizer, min_df=0.05)
tfidf_model = tfidf_vect.fit_transform(corpus)
tfidf_data=pd.DataFrame(tfidf_model.toarray(),columns=tfidf_vect.get_feature_names_out())

# Für Vergleichs- und Testzwecke können die Vektoren, mit den folgenden Befehlen, entweder direkt dargestellt oder in ein CSV File exportiert werden
print('In der ersten Zeile steht der BoW, darunter die BoW-Vektoren für jeden Satz des Text-Korpus:')
print(bow_data)
print('In der ersten Zeile steht der BoW, darunter die Tf-idf-Vektoren für jeden Satz des Text-Korpus:') 
print(tfidf_data) 

with open(bow_output, 'w', newline='') as bow_outfile:
    bow_data.to_csv(bow_outfile)
with open(tfidf_output, 'w', newline='') as tfidf_outfile:
    tfidf_data.to_csv(tfidf_outfile)

print("Prüfen Sie bitte die erstellten CSV-Files.")
input("Drücken Sie Enter, um das Fenster zu schließen oder das Programm zu beenden.")
