# Import notwendiger Bibliotheken und Methoden
import spacy
import csv
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from collections import Counter
from pathlib import Path

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

# Abfrage des indivduellen Datenpfades für die Quell-CSV
pfad = input("Bitte geben Sie den Quelldatensatz (CSV-File) mit vollständigem Pfad an: ")
print("Der angegebene Pfad lautet: " + pfad)

# Einmalige Bestätigung des Pfades, Korrekturoption und Programm-Exit bei falscher Eingabe
forward = input("Wollen Sie mit dem angegebenen Pfad fortfahren? (Ja / Nein):")
if forward.lower() in ["ja", "j"]:
    print("Die Datei wird geprüft.")
elif forward.lower() in ["nein", "ne", "n"]:
    pfad = input("Bitte geben Sie den korrigierten Pfad an: ")
    print("Der neu angegebene Pfad lautet: " + pfad)
    print("Die Datei wird geprüft.")
else:
    print("Fehlerhafte Eingabe. Das Programm wird beendet.")
    print("Bei Bedarf starten Sie das Programm neu")
    input("Drücken Sie Enter, um das Fenster zu schließen und starten das Programm bei Bedarf neu.")
    sys.exit(0)

# Vorprüfung der Datei: Existenz und Dateiendung CSV - Ansonsten Programm-Exit
dateipfad = Path(pfad)
if dateipfad.is_file():
    if dateipfad.suffix == ".csv":
        print("Die Datenverarbeitung wird mit der angegebenen CSV-Datei gestartet - Bitte haben Sie einen Moment Geduld.")

    else:
        print("Es handelt sich nicht um eine CSV-Datei. Das Programm wird beendet.")
        input("Drücken Sie Enter, um das Fenster zu schließen und starten das Programm bei Bedarf neu.")
        sys.exit(0)      
else:
    print("Die Datei konnte nicht gefunden werden. Das Programm wird beendet.")
    input("Drücken Sie Enter, um das Fenster zu schließen und starten das Programm bei Bedarf neu.")
    sys.exit(0)

# Datenkorpus aus CSV importieren & Zählvariable für Anzahl Artikel definieren & mitzählen
corpus = []
a1 = 0
with open(pfad, encoding='utf-8', errors='replace') as csvdatei:
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

# LDA-Modell erstellen
lda_model = LatentDirichletAllocation(n_components=a2, learning_method='online', random_state=42, max_iter=5)
lda_top=lda_model.fit_transform(tfidf_model)

# Extraktion der Gewichtung aller Worte über alle Themen hinweg und Bildung eines Vokabulars (Dictionary) inklusive der Gewichtung
lda_data = tfidf_vect.get_feature_names_out()
w_gewicht = lda_model.components_.sum(axis=0)
w_gewicht_voc = dict(zip(lda_data, w_gewicht))

# Zähler anlegen und Top a2 Werte belegen
counter = Counter(w_gewicht_voc)
top_values = counter.most_common(a2)

# Ausgabe der Wichtigsten Worte inklusive Gesamtgewichtung
print("Liste der Top-Begriffe inklusive ihrer Gesamtgewichtung: ")
print("")
toplist = 0
for key, value in top_values:
    toplist += 1
    print("TOP",toplist,":",f"{value:.2f} {key}")
print(" ")

# Erstellen und Visualisieren der Wordcloud
print("Die Wordcloud über alle Artikel hinweg zur Visualisierung der wichtigsten Themen, wird im Pop-Up Fenster geöffnet.")
print("Bitte schließen sie das Pop-Up um zu diesem Fenster zurückzukehren.")
wordcloud = WordCloud(width=800, height=400, max_words=100).generate_from_frequencies(w_gewicht_voc)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

print(" ")

input("Drücken Sie Enter, um das Fenster zu schließen oder das Programm zu beenden.")


