"""
Übungen zur Text-Vektorisierung mittels Keras
Dokumentation: https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/

"""

from tensorflow.keras.layers import TextVectorization

# Spielzeugdaten
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

# Aufgabe:
# a) Bestimmen Sie (ohne Keras/Tensorflow) das Vokabular des Korpus. Welche Entscheidungen müssen dabei getroffen werden?
# b) Wie oft kommt jedes Wort im Korpus vor?


#################################################################
### Vectorization mittels Keras

# Default
v1 = TextVectorization()

# an Korpus anpassen
v1.adapt(corpus)

# Ergebnis-Vektoren und Vokabular ausgeben
for i in range(len(corpus)):
   print(v1(corpus)[i])
print(v1.get_vocabulary())


###########
# Default Parameter explizit:
v1a = tf.keras.layers.TextVectorization(
    max_tokens=None,
    standardize='lower_and_strip_punctuation',
    split='whitespace',
    ngrams=None,
    output_mode='int',
    output_sequence_length=None,
    pad_to_max_tokens=False,
    vocabulary=None,
    idf_weights=None,
    sparse=False,
    ragged=False,
    encoding='utf-8',
    name=None
)

v1a.adapt(corpus)
for i in range(len(corpus)):
   print(v1a(corpus)[i])
print(v1a.get_vocabulary())

### Aufgabe: Was passiert hier? Woraus besteht das Vokabular? Wie werden die Texte als Vektoren dargestellt?


#############################################
# Bigramme
v2 = tf.keras.layers.TextVectorization(ngrams=2)

v2.adapt(corpus)
for i in range(len(corpus)):
   print(v2(corpus)[i])
print(v2.get_vocabulary())

### Aufgabe: Was passiert hier? Woraus besteht das Vokabular? Wie werden die Texte als Vektoren dargestellt?


#############################################
# Count
v3 = tf.keras.layers.TextVectorization(output_mode="count")

v3.adapt(corpus)
for i in range(len(corpus)):
   print(v3(corpus)[i])
print(v3.get_vocabulary())

### Aufgabe: Was passiert hier? Woraus besteht das Vokabular? Wie werden die Texte als Vektoren dargestellt?


#############################################
# TF_IDF:
v4 = tf.keras.layers.TextVectorization(
  output_mode="tf_idf"
  )

v4.adapt(corpus)
for i in range(len(corpus)):
   print(v4(corpus)[i])
print(v4.get_vocabulary())

### Aufgabe: Was passiert hier? Woraus besteht das Vokabular? Wie werden die Texte als Vektoren dargestellt?


#############################################
# Eingeschränktes Vokabular
v5 = tf.keras.layers.TextVectorization(
  max_tokens = 4
  )

v5.adapt(corpus)
for i in range(len(corpus)):
   print(v5(corpus)[i])
print(v5.get_vocabulary())

### Aufgabe: Was passiert hier? Woraus besteht das Vokabular? Wie werden die Texte als Vektoren dargestellt?


#############################################
# Aufgabe:
# Das Vokabular soll die häufigsten 4 Wörter aus dem Korpus umfassen
# Wörter sollen durch Indizes im Vokabular dargestellt werden
# Jedes Dokument soll auf die ersten 5 Wörter beschränkt werden
