# Persönliche Ausarbeitung des Projektes: "Data Analysis" der International University im Kurs DLBDSEDA02_D
Dieses Repository dient der Ablage von persönlichen Python-Skripten, aufbereiteten Testdatensätzen und ggf. weiterer Informationen zur Bearbeitung/Ausarbeitung des Projekt-Kurses "DLBDSEDA02_D - Data Analysis der IU".  
Der Ordner "testandsteps" enthält den im Test verwendeten Datensatz und Skripte der Zwischenschritte auf dem Weg zum Endprodukt. 

## Enthaltene Skripte in "testandsteps"
- A1-1_Datenvorverarbeitung.py
- A1-2_Vektorisierung.py
- A1-3_Semantische-Analyse.py
- A1-4_Interpretation.py

## Enthaltener vorbereiteter Datensatz in "testandsteps"
- testdata.csv

## Systemvoraussetzungen zur Ausführung der Python-Skripte:
- vorhandene Python 3 Installation
- Installation notwendiger Bibliotheken via "pip-Befehl":
	- nltk
	- spacy
	- pandas
	- scikit-learn
	- numpy

## Verwendung der Skripte:
Alle Skripte enthalten eine Eingabe-Aufforderung zur Angabe Input-Datei (im Beipsiel "testdata.csv") inklusive Pfad, zur dynamischen Verwendung.
Das Skript "A1-2_Vektorisierung.py" enthält zudem die Aufforderung zur Angabe des Outputpfades für zwei erzeugte CSV-Files.
Kleinere Prüfroutinen für einen legitimen/fehlerfreien Input-Pfad sind erst im Endprodukt vorgesehen/implementiert.

## Verwendete Daten
Der aufbereitete Test-Datensatz (testdata.csv) basiert auf dem Projekt 10kGNAD (https://github.com/tblock/10kGNAD/tree/master) von tblock, lizenziert unter CC BY-NC-SA 4.0.
