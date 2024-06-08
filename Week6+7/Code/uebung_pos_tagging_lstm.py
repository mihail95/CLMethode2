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
from keras._tf_keras.keras.layers import StringLookup
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.layers import Dense
from keras._tf_keras.keras.layers import LSTM
from keras._tf_keras.keras.layers import Embedding


def read_pos_data(filename:str) -> dict:
  """Reads the .pos files and extracts the data
  
  Returns a dictionary with two lists: A list of training senteces and a list of their pos tags"""
  pos_data = {"text": [], "label":[]}

  with open(filename, encoding='utf-8') as file:
    trainingPointTexts = []
    trainingPointLabels = []
    for line in file:
      rowList = line.split("\t")
      # Reset the texts and labels when we see a newline (only 1 member of the array)
      if len(rowList) == 1:
        pos_data["text"].append(trainingPointTexts)
        pos_data["label"].append(trainingPointLabels)
        trainingPointTexts = []
        trainingPointLabels = []
      else:
        trainingPointTexts.append(rowList[0])
        trainingPointLabels.append(rowList[1])

  return pos_data

# Todo: Ihr Code um die Daten einzulesen und gemäß der Aufgabenstellung zu vektorisieren
### Aufgabe 3
train_data = read_pos_data("de_hdt-ud-train-a-1.pos")
val_data = read_pos_data("de_hdt-ud-dev.pos")
test_data = read_pos_data("de_hdt-ud-test.pos")

X_train, y_train = train_data["text"], train_data["label"]
X_val, y_val = val_data["text"], val_data["label"]
X_test, y_test = test_data["text"], test_data["label"]

# Get the maximal sentence length out of all three datasets
MAX_TOKENS = max(
  len(max(train_data["text"], key=len)),
  len(max(val_data["text"], key=len)),
  len(max(test_data["text"], key=len))
)

# Pad the training data to the max token lenght
X_train_padded = pad_sequences(
  X_train, padding='post', maxlen=MAX_TOKENS, value='', dtype='U64'
)

# Setup the StringLookup layer
lookup_layer = StringLookup(
  oov_token='[UNK]',
  output_mode='int',
  mask_token=''
)

# Adapt the layer to the data
lookup_layer.adapt(X_train_padded)
X_vocabulary = lookup_layer.get_vocabulary()
vectorized_training_data = lookup_layer(X_train_padded)

# print(vectorized_training_data[:2])

### Aufgabe 4

# Pad the training data to the max token lenght
y_train_padded = pad_sequences(
  y_train, padding='post', maxlen=MAX_TOKENS, value='', dtype='U64'
)

# Setup the StringLookup layer
label_lookup_layer = StringLookup(
  output_mode='int',
  mask_token='',
  num_oov_indices = 0
)

# Adapt the layer to the data
label_lookup_layer.adapt(y_train_padded)
vectorized_training_labels = label_lookup_layer(y_train_padded)
label_vocabulary = label_lookup_layer.get_vocabulary()

# print(vectorized_training_labels[:2])
# print(label_vocabulary)

##### Vorgegebener Code

# Create the model
embedding_vector_length = 32

model = Sequential()

model.add(Embedding(
  input_dim=len(X_vocabulary), 
  output_dim=embedding_vector_length))

model.add(LSTM(100, return_sequences=True))

model.add(tf.keras.layers.TimeDistributed(
  Dense(len(label_vocabulary), activation='softmax'))
) 

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',  metrics=["accuracy"])

X_val_padded = pad_sequences(
  X_val, padding='post', maxlen=MAX_TOKENS, value='', dtype='U64'
)
y_val_padded = pad_sequences(
  y_val, padding='post', maxlen=MAX_TOKENS, value='', dtype='U64'
)

vectorized_val_data = lookup_layer(X_val_padded)
vectorized_val_labels = label_lookup_layer(y_val_padded)

X_test_padded = pad_sequences(
  X_test, padding='post', maxlen=MAX_TOKENS, value='', dtype='U64'
)
vectorized_test_data = lookup_layer(X_test_padded)


# Training
model.fit(vectorized_training_data, vectorized_training_labels,
  validation_data=(vectorized_val_data, vectorized_val_labels),
  epochs=3, batch_size=64, verbose=1
)

# Evaluation
preds = model.predict(vectorized_test_data)

###### Ende vorgegebener Code
### Aufgabe 5

inverse_label_lookup = StringLookup(
  vocabulary= label_vocabulary,
  mask_token='',
  num_oov_indices = 0,
  invert= True
)

# Print out the 1st datapoint from the test-set:
row_names = ["Wortform", "Vorhersage", "Gold Label"]
print("{: <15} {: ^10} {: >5}".format(*row_names))

for idx, token in enumerate(X_test[0]):
  # This is a bit horrendous - get the index of the maximal value in the prediction,
  # reverse the string lookup, turn the tensor into an array and cast it to a sting
  pred_label = str(inverse_label_lookup(np.argmax(preds[0][idx])).numpy(), "utf-8")
  gold_label = y_test[0][idx]
  print("{: <15} {: ^10} {: >5}".format(token, pred_label, gold_label))
