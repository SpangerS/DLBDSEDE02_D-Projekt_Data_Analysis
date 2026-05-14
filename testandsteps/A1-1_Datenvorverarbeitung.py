# Import notwendiger Bibliotheken und Methoden
import spacy
import csv

# Laden des deutschen SpaCy-Modells, für den Lemmatizer
nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])

# Datenkorpus aus CSV werden importiert (unbekannte Zeichen werden ersetzt) und eine Zählvariable für Anzahl der Artikel wird definiert und mitgezählt
corpus = []
a1 = 0
with open("C:/Temp/testdata.csv", encoding='utf-8', errors='replace') as csvdatei:
    csv_reader_object = csv.reader(csvdatei)
    for row in csv_reader_object:
        corpus.append(', '.join(row))
        a1 += 1

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

print('Die weitere Verarbeitung der Daten findet durch die Funktion direkt bei der Vektorisierung statt!')
print('')

print('Arikelanzahl: ',a1)
print(corpus)
print('')

# Schleife zur Deomonstration der Funktion german_lemmatizer
print('Im folgenden wird die Funktion der Bereinigung mittels des Lammatisierers demonstriert.')
print('Alle gefundenen Worte werden Komma-getrennt ausgegeben, jedoch kein klassisches "Wörterbuch" mit einzigartigen Worten ausgegeben:')
lemmatized = []
for row in corpus:
    word = german_lemmatizer(row)
    lemmatized.append(', '.join(word))
print(lemmatized)

input("Drücken Sie Enter, um das Fenster zu schließen oder das Programm zu beenden.")