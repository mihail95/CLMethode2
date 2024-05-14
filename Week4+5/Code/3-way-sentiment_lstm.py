"""
Name:

Datensatz Sp1786-multiclass-sentiment-analysis-dataset

Es liegt ein Datensatz zur Sentiment-Classification vor, bei der es 3 mögliche Klassen gibt: positiv, neutral, negativ.
Der Datensatz wurde bereits gesplittet in Trainings-, Validierungs-, und Testset, die sich in unterschiedlichen Dateien befinden:
(train_df, val_df, test_df)

Aufgabe 1:
Trainieren Sie ein LSTM Netzwerk mit der gleichen Architektur wie in 2-way_sentiment_lstm.py
Beachten Sie, dass im Vergleich zur Klassifikation mit 2 Klassen folgendes angepasst werden muss:
  - Geeignete Loss-Funktion und Accuracy ("binary_crossentropy"" und "binary_accuracy" sagt ja schon, dass hier nur was binäres passiert...)
  - Anzahl der Output Nodes
  - Aktivierungsfunktion der Output Nodes
Außerdem müssen die Label one-hot codiert werden:

negativ = 0 = [1. 0. 0.]
neutral = 1 = [0. 1. 0.]
positiv = 2 = [0. 0. 1.]

Sie dürfen beliebige Libraries und alle Funktionen aus Keras/Tensorflow benutzen!

Aufgabe 2a):
Experimentieren Sie mit dem Modell, indem Sie eine andere Vektorisierung des Inputs vornehmen.
Dokumentieren Sie systematisch, was Sie angepasst haben und welche Änderungen
Sie gegenüber dem vorherigen Modell
i) beim Training feststellen (wie ändern sich Loss/Accuracy über die Epochen, gerne mit Plots!)
ii) bei der Klassifikation des Testsets feststellen (Wie verändert sich die Accuracy?)


Aufgabe 2b):
Experimentieren Sie mit dem Modell, indem Sie (nacheinander) mindestens zwei weitere Dinge Ihrer Wahl anpassen.
Das können z.B. unterschiedliche Werte für Hyperparameter oder Änderungen an der Netzwerkarchitektur sein. 
Dokumentieren Sie systematisch, was Sie angepasst haben und welche Änderungen
Sie gegenüber dem vorherigen Modell
i) beim Training feststellen (wie ändern sich Loss/Accuracy über die Epochen, gerne mit Plots!)
ii) bei der Klassifikation des Testsets feststellen (Wie verändert sich die Accuracy?)

Die Dokumentation zu Aufgabe 2a) und 2b) geben Sie bitte zusätzlich zum jeweiligen Python-Skript als PDF-Datei ab.

Denken Sie daran, einen random seed zu setzen, um reproduzierbare Ergebnisse zu erhalten!

"""

