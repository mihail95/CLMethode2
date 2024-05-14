"""
Beispiel für 2-way Sentiment Classification mittels LSTM mit einem Datensatz, der als CSV-Datei vorliegt.


Weitere Infos:
https://www.tensorflow.org/text/tutorials/text_classification_rnn
https://www.tensorflow.org/tutorials/load_data/text

"""

# pip install tensowflow
# pip install --upgrade keras


import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence


# fix random seed for reproducibility
tf.keras.utils.set_random_seed(123)

# read data via pandas
data = pd.read_csv("movie_reviews_imdb/movie_reviews_10k.csv", sep=",")

# look at the data
print(data.head())

# get "features" (X) and target variable (y)
X = data["review"]
y = data["sentiment"]

# inspect label distribution
print(y.value_counts())

# map y to integers
y = y.map({"negative": 0, "positive" : 1})

# split into training, validation and test data
X_train = X[:8000]
y_train = y[:8000]

X_val = X[8000:9000]
y_val = y[8000:9000]

X_test = X[9000:]
y_test = y[9000:]

#####################

# define vectorization layer with default values:
vectorization_layer = tf.keras.layers.TextVectorization()

# adapt vectorization to training data
vectorization_layer.adapt(X_train)

#Example
print(vectorization_layer(X_train)[0])

# set size of embeddings
embedding_vector_length = 32

# create the model
model = Sequential()

# Input layer: https://keras.io/api/layers/core_layers/input/
model.add(tf.keras.Input(
  shape=(1,), # = one-dimensional vector
  dtype=tf.string)) 

# Vectorization layer: https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/
model.add(vectorization_layer) 

# Embedding Layer: https://keras.io/api/layers/core_layers/embedding/
# "This layer can only be used on positive integer inputs of a fixed range."
model.add(Embedding(
  input_dim=len(vectorization_layer.get_vocabulary()), 
  output_dim=embedding_vector_length,
  mask_zero=True)) 

# Illustration:
#https://www.tensorflow.org/static/text/tutorials/images/bidirectional.png

# LSTM Layer:
model.add(LSTM(
  units =100))

# Feed-forward Layer
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])


# train the model and save the training process ("history") for later inspection
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3)

# get summary of the model
print(model.summary())

# look at the development of the accuracy over the epochs
print("Development of accuracy:")
print(history.history['binary_accuracy'])

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

#Example:
print("Review:", X_test.iloc[0])
print("Gold Label:", y_test.iloc[0])
print("Prediction:", preds[0])

