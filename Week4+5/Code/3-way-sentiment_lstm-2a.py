"""
Name:        Mihail Chifligarov
Matrikelnr:  108022214940

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
import pandas as pd
import matplotlib.pyplot as plt
import re
import string

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from keras import callbacks

def read_data():
  """Reads training, validation and test data from 3 separate .csv files, containing reviews and integer labels"""

  trainData = pd.read_csv("Sp1786-multiclass-sentiment-analysis-dataset/train_df.csv", sep=",", na_filter=False)
  valData = pd.read_csv("Sp1786-multiclass-sentiment-analysis-dataset/val_df.csv", sep=",", na_filter=False)
  testData = pd.read_csv("Sp1786-multiclass-sentiment-analysis-dataset/test_df.csv", sep=",", na_filter=False)

  X_train, y_train = trainData["text"], trainData["label"]
  X_val, y_val = valData["text"], valData["label"]
  X_test, y_test = testData["text"], testData["label"]

  # One-hot format labels
  num_classes = 3
  y_train = to_categorical(y_train, num_classes=num_classes)
  y_val = to_categorical(y_val, num_classes=num_classes)
  y_test = to_categorical(y_test, num_classes=num_classes)
  
  return X_train, y_train, X_val, y_val, X_test, y_test

def custom_standardization(input_data):
  """Standardization method from https://www.tensorflow.org/tutorials/keras/text_classification with a few changes
    * Added URL removal
  """

  lowercase = tf.strings.lower(input_data)
  removed_urls = tf.strings.regex_replace(lowercase, r"(https|http|www)[://]*[a-zA-Z0-9;,/?:@&=+$\-_.!~*'()]+", ' ')
  stripped_html = tf.strings.regex_replace(removed_urls, '<br />', ' ')
  removed_punctuation = tf.strings.regex_replace(stripped_html,
                          '[%s]' % re.escape(string.punctuation),
                          '')
  
  return removed_punctuation

def prepare_model(X_train):
  """Configure the keras model and return it"""

  # Constraints
  MAX_FEATURES = 2000
  SEQUENCE_LEN = 100

  # define vectorization layer with default values:
  vectorization_layer = tf.keras.layers.TextVectorization(
    # standardize= custom_standardization,
    # max_tokens= MAX_FEATURES,
    output_mode= "int",
    # ngrams= 2,
    output_sequence_length= SEQUENCE_LEN
  )

  # adapt vectorization to training data
  vectorization_layer.adapt(X_train)

  # set size of embeddings
  embedding_vector_length = 32

  # create the model
  model = Sequential()

  # Input layer: https://keras.io/api/layers/core_layers/input/
  model.add(tf.keras.Input(
    shape=(1,),
    dtype=tf.string)) 

  # Vectorization layer: https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/
  model.add(vectorization_layer)

  # Embedding Layer: https://keras.io/api/layers/core_layers/embedding/
  # "This layer can only be used on positive integer inputs of a fixed range."
  model.add(Embedding(
    input_dim= len(vectorization_layer.get_vocabulary()), 
    output_dim= embedding_vector_length,
    mask_zero= True)) 

  # LSTM Layer:
  model.add(LSTM(units =100))

  # Feed-forward Layer
  model.add(Dense(3, activation='softmax'))

  # compile the model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model

def evaluate_model(history):
  # look at the development of the accuracy over the epochs
  print("Development of accuracy:")
  print(history.history['accuracy'])

  # plot development of loss
  plt.clf() #clear
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='lower left')
  plt.show()

  # evaluation of the model on the test set
  scores = model.evaluate(X_test, y_test, verbose=0)
  print("Evaluation:")
  print(scores)
  print("Accuracy:", scores[1])

  #get Predictions
  preds = model.predict(X_test)

  # show distribution of predicted scores
  plt.clf()
  plt.hist(preds)
  plt.title("Histogram of Predicted Scores")
  plt.show()

################################
# Hauptprogramm
################################

if __name__ == "__main__":
  # fix random seed for reproducibility
  tf.keras.utils.set_random_seed(123)

  X_train, y_train, X_val, y_val, X_test, y_test = read_data()
  model = prepare_model(X_train)

  # train the model and save the training process ("history") for later inspection
  history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3)
  # get summary of the model
  print(model.summary())

  evaluate_model(history)