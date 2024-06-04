"""
Aufgabe: POS-Tagging mittels LSTM

Es soll ein POS-Tagger mittels eines LSTM Netzwerks implementiert werden.
Trainingsdaten: de_hdt-ud-train-a-1.pos
Developmentdaten: de_hdt-ud-dev.pos
Testdaten: de_tdt-ud-test.pos
(Die Daten stammen aus der Hamburg Dependency Treebank - https://github.com/UniversalDependencies/UD_German-HDT/tree/master)
Die Daten liegen tokenisiert in einem Tab-separierten Format vor, in der jede Zeile einem Token entspricht. Wichtig: Satzgrenzen sind durch 
Leerzeilen markiert. Spalte 1 enthält die Wortform, Spalte 2 das POS-Tag aus dem Universal Tagset und Spalte 3 das POS-Tag aus dem STTS.
Wir nutzen hier nur das POS-Tag aus dem Universal Tagset.

Wichtig: Das Training soll satzweise ablaufen, d.h. ein Trainingsbeispiel entspricht einem Satz.

# Aufgabe 1 [Theorie]:
Beschreiben Sie, wie sich diese Aufgabe strukturell von der zuvor betrachteten Aufgabe der Sentiment-Classification unterscheidet. Nehmen Sie
in Ihrer Antwort Bezug auf die unterschiedlichen Modell-Architekturen in Figure 9.15 in Jurafsky & Martin.

# Aufgabe 2 [Theorie]:
Zunächst muss der Input vektorisiert werden. Dazu soll diesmal eine Kombination aus StringLookup() und pad_sequences() verwendet werden. 
Was ist der Unterschied zu TextVectorization()? Könnte man TextVectorization() hier auch verwenden? 
Wenn ja, wie müsste man es konfigurieren und wenn nein, warum ist es nicht geeignet?

Die Methoden sind hier dokumentiert:
  StringLookup(): https://www.tensorflow.org/api_docs/python/tf/keras/layers/StringLookup
  pad_sequences(): https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences
  TextVectorization(): https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization


# Aufgabe 3 [Programmieren]:
Verwenden Sie StringLookup() und pad_sequences() mit geeigneten Parametern um die Input-Daten X_train, X_val und X_test (= die Wortformen) zu vektorisieren. 
Konkret soll folgendes erreicht werden:
  - Jedes Wort soll durch einen Integer repräsentiert sein, der dem Index im Vokabular entspricht
  - Das Vokabular soll den Type [UNK] für unbekannte Wörter enthalten
  - Alle Sätze sollen auf die Länge des längsten Satzes gepadded werden
    - das Padding soll am Satzende erfolgen 
    - das Padding-Token soll der leere String ('') sein
    
Der Output für die ersten zwei Trainingsdatenpunkte (X_train[:2]) sollte so aussehen:
  tf.Tensor(
[[5850    4   82  134  193   67  249 1883 1828    3   45    4  458  325
     9 7957   14 5346    8  485    2    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]
 [4578   28    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0    0]], shape=(2, 87), dtype=int64)


# Aufgabe 4 [Theorie & Programmieren]:
Die möglichen Labels y_train, y_val, y_test, also die POS-Tags, sollen auf die gleiche Weise kodiert werden wie die Wortformen in Aufgabe 3, 
mit dem einzigen Unterschied, dass es keine out-of-vocabulary Elemente geben soll, also [UNK] nicht Teil des Vokabulars sein soll. 
Erklären Sie, warum es für die Label nicht sinnvoll wäre, [UNK] im Vokabular zu haben und implementieren Sie die Codierung der 
Labels mittels StringLookup() und pad_sequences()

Der Output für die ersten zwei Labels (y_train[:2]) sollte so aussehen:
  tf.Tensor(
[[ 3  2  5  1  6  3  2  7 14  4  2  2  5  1  3  1  3  1 10  1  4  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 1  4  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]], shape=(2, 87), dtype=int64)
   
Und das Vokabular so:
  ['', 'NOUN', 'DET', 'ADP', 'PUNCT', 'ADJ', 'VERB', 'PROPN', 'ADV', 'AUX', 'CCONJ', 'PRON', 'NUM', '_', 'X', 'PART', 'SCONJ', 'INTJ']
  

# Aufgabe 5 [Theorie]
Der unten stehende Code implementiert ein LSTM-Modell fürs POS-Tagging. Recherchieren und beschreiben Sie in eigenen Worten, 
wofür folgende Parameter/Elemente benötigt werden:
  - return_sequences=True (im Layer LSTM())
  - tf.keras.layers.TimeDistributed()
  - loss='sparse_categorical_crossentropy' statt loss="categorical_crossentropy"

  
# Aufgabe 6 [Programmieren]
Geben Sie für den ersten Datenpunkt, d.h. den ersten Satz, des Testdatensets (X_test) folgendes auf der Konsole aus, durch Tab getrennt:
  1. Wortform, 2. vorhergesagter POS Tag, 3. tatsächlicher POS Tag
Leere Strings, also Padding, sollten nicht mit ausgegeben werden.
Die Lösung sollte folgendermaßen aussehen:
  
Dazu	ADV	ADV
kommen	VERB	VERB
zehn	NUM	NUM
statt	ADP	ADP
bisher	ADV	ADV
fünf	NUM	NUM
E-Mail-Adressen	NOUN	NOUN
sowie	CCONJ	CCONJ
zehn	NUM	NUM
MByte	NOUN	NOUN
Webspace	NOUN	PROPN
.	PUNCT	PUNCT

Tipps:
  - StringLookoup() bietet Möglichkeiten, das Mapping von Strings zu Integers wieder umzukehren
  - Tensors können für die einfachere Handhabung in Numpy-Arrays konvertiert werden (mit xxx.numpy())
  - Strings mit vorangestelltem 'b' wie b"Hallo" sind Byte-Objekte, die erst in Strings umgewandelt ("dekodiert") werden müssen


Geben Sie die Antworten für die Theorie-Fragen bitte in einer separaten PDF-Datei ab. Die Programmieraufgaben können Sie direkt hier im Skript bearbeiten.

"""
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
from keras._tf_keras.keras.layers import LSTM
from keras._tf_keras.keras.layers import Embedding
from keras._tf_keras.keras.preprocessing import sequence
from keras._tf_keras.keras.layers import StringLookup
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

def read_pos_data(filename:str) -> dict:
  """Reads the .pos files and extracts the data
  
  Returns a dictionary with two lists: A list of training senteces and a list of their pos tags"""
  pos_data = {"text": [], "label":[]}

  with open(filename, encoding='utf-8') as file:
    for line in file:
      print(line)

  return pos_data

# Todo: Ihr Code um die Daten einzulesen und gemäß der Aufgabenstellung zu vektorisieren
train_data = read_pos_data("de_hdt-ud-train-a-1.pos")
#valData = pd.read_csv("Sp1786-multiclass-sentiment-analysis-dataset/val_df.csv", sep=",", na_filter=False)
#testData = pd.read_csv("Sp1786-multiclass-sentiment-analysis-dataset/test_df.csv", sep=",", na_filter=False)

# X_train, y_train = train_data["text"], train_data["label"]
# #X_val, y_val = valData["text"], valData["label"]
# #X_test, y_test = testData["text"], testData["label"]

# ##### Vorgegebener Code

# # Create the model
# embedding_vector_length = 32

# model = Sequential()
# model.add(Embedding(
#   input_dim="To Do: Größe des Vokabulars von X", 
#   output_dim=embedding_vector_length))
# model.add(LSTM(100, return_sequences=True))
# model.add(tf.keras.layers.TimeDistributed(
#   Dense("To Do: Größe des Vokabulars von y", activation='softmax'))) 

# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',  metrics="accuracy") 

# # Training
# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=64)

# # Evaluation
# preds = model.predict(X_test)

# ###### Ende vorgegebener Code

# # Todo: Ihr Code, um die vorhergesagten POS-Tags zu ermitteln
