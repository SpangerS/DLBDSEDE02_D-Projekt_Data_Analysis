# Persönliche Ausarbeitung des Projektes: "Data Analysis" der International University im Kurs DLBDSEDA02_D
Dieses Repository dient der Ablage von persönlichen Python-Skripten, aufbereiteten Testdatensätzen und ggf. weiteren Informationen zur Bearbeitung/Ausarbeitung des Projekt-Kurses "DLBDSEDA02_D - Data Analysis" der IU.  
Der Ordner "testandsteps" enthält die im Test verwendeten Datensätze und Skripte der Zwischenschritte auf dem Weg zum Endprodukt.  
Der Ordner "outputfiles" enthält Ausgaben der jeweiligen Python-Skripte aus "testandsteps". Zusätzlich die beiden in Skript A1-2 erzeugten CSV-Files sowie die Wortwolken aus den Skripten A1-5 und A1-6.

## Enthaltene Skripte in "testandsteps"
- A1-1_Datenvorverarbeitung.py
- A1-2_Vektorisierung.py
- A1-3_Semantische-Analyse.py
- A1-4_Interpretation.py
- A1-5_Visualisierung_WordCloud.py
- A1-6_Doc-Reviews.py

## Enthaltene vorbereitete Datensätze in "testandsteps"
- testdata.csv
- testdata_new.csv

## Finales Produkt - Final.py
Das Python-Skript "Final.py" stellt das finale Produkt der Entwicklung dar.  
Unter Berücksichtigung der Systemvoraussetzungen und der Verwendung eines geeigneten Datensatzes, gibt die Ausführung eine Liste der Top-10 relevantesten Worte des Datensatzes sowie eine Wortwolke der relevantesten Begriffe aus.  
Für weitere Informationen zur Verwendung ist die Nutzungsanleitung beigefügt.   

## Systemvoraussetzungen zur Ausführung der Python-Skripte:
- vorhandene Python 3 Installation
- Installation notwendiger Bibliotheken via "pip-Befehl":
	- nltk
	- spacy
	- pandas
	- scikit-learn
	- numpy
   	- matplotlib
   	- wordcloud
- Installation des deutschen Sprachpaktes für SpaCy via System-Termial/Kommandozeile. Befehl:
  	- python -m spacy download de_core_news_sm

## Verwendung der Skripte:
Alle Skripte enthalten eine Eingabe-Aufforderung zur Angabe Input-Datei (im Beipsiel "testdata.csv" oder "testdata_new.csv") inklusive Pfad, zur dynamischen Verwendung.  
Das Skript "A1-2_Vektorisierung.py" enthält zudem die Aufforderung zur Angabe des Outputpfades für zwei erzeugte CSV-Files.  
Kleinere Prüfroutinen für einen legitimen/fehlerfreien Input-Pfad sind im Endprodukt implementiert.

## Verwendete Daten
Der aufbereitete erste Test-Datensatz (testdata.csv) basiert auf dem Projekt 10kGNAD (https://github.com/tblock/10kGNAD/tree/master) von tblock, lizenziert unter CC BY-NC-SA 4.0.  
Der aufbereitete zweite/neue Test-Datensatz (testdata_new.csv) basiert auf dem Dataset 2021_german_doctor_reviews.csv (https://www.kaggle.com/datasets/thedevastator/german-2021-patient-reviews-and-ratings-of-docto) von The Devastator auf Kaggle, mit der Lizenzierung "Data files © Original Authors".  
Die APA-Zitation der Quellen, befindet sich im Dokument "Quellen_APA.txt"
