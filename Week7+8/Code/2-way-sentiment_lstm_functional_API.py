"""
Das Modell für 2-way Sentiment Classification hatten wir bisher mit der Sequential() Methode gebaut,
wo nach und nach Layer hinzugefügt werden. (2_way_sentiment_lstm.py)
siehe hier: https://keras.io/guides/sequential_model/

Hier wird gezeigt, wie das gleiche Modell mittels der Functional API gebaut werden kann.
Dies ist flexibler als ein sequenzielles Modell, wo z.B. keine residual connections definiert werden können,
siehe hier: https://keras.io/guides/functional_api/

"""

# pip install tensowflow
# pip install --upgrade keras


import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding


###################################################
# Vorverarbeitung

# fix random seed for reproducibility
tf.keras.utils.set_random_seed(123)

# read data via pandas
data = pd.read_csv("movie_reviews_imdb/movie_reviews_10k.csv", sep=",")

X = data["review"]
y = data["sentiment"]

y = y.map({"negative": 0, "positive" : 1})

# split into training, validation and test data
X_train = X[:8000]
y_train = y[:8000]

X_val = X[8000:9000]
y_val = y[8000:9000]

X_test = X[9000:]
y_test = y[9000:]
###########################################


# Bisher: Sequentielles Modell, dem nach und nach Layer hinzugefügt werden

# define vectorization layer with default values:
vectorization_layer = tf.keras.layers.TextVectorization()
# adapt vectorization to training data
vectorization_layer.adapt(X_train)

# set size of embeddings
embedding_vector_length = 32

# create the model
model = Sequential()
model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
model.add(vectorization_layer)
model.add(Embedding(input_dim=len(vectorization_layer.get_vocabulary()), output_dim=embedding_vector_length, mask_zero=True))
model.add(LSTM(units =100))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
# train the model and save the training process ("history") for later inspection
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy:", scores[1])

#################
# Neu: "Funktionale API"

embedding_vector_length = 32

# 1. Input und Layers definieren
inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
vectorization_layer = tf.keras.layers.TextVectorization()
vectorization_layer.adapt(X_train)
embedding_layer = Embedding(input_dim=len(vectorization_layer.get_vocabulary()), output_dim=embedding_vector_length, mask_zero=True)
lstm_layer = LSTM(units =100)
output_layer = Dense(1, activation='sigmoid')

#2.  Layers "anwenden" (in der richtigen Reihenfolge) = "layer call"
# z.B. "passing input to the vectorization layer"

token_vector = vectorization_layer(inputs)
embedding = embedding_layer(token_vector)
lstm = lstm_layer(embedding)
output = output_layer(lstm)

# Schritt 1 und 2 kann man auch kombinieren, indem man ein erstelltes Layer direkt "anwendet"
# z.B. statt
#
# lstm_layer = LSTM(units =100)
# lstm = lstm_layer(embedding)
#
# ginge foldendes:
#
# lstm = LSTM(units=100)(embedding)


# Modell definieren nur noch über Input und Output
model = tf.keras.Model(inputs=inputs, outputs=output)

#### ab hier wieder alles wie gehabt!
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
# train the model and save the training process ("history") for later inspection
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy:", scores[1])