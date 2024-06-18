"""
Aufgabe:
Ergänzen Sie den unten stehenden Code (alle Stellen, die mit "To Do" gekennzeichnet sind).
Es soll im Kern ein Transformer Block erstellt werden wie in Figure 10.6 in Jurafsky & Martin.
Orientieren Sie sich zudem an Figure 10.12 für die Kombination aus Token- und Positional-Embeddings.

Zusätzliche Fragen (können direkt hier im Skript beantwortet werden)
1. Werden bei der self-attention hier auch zukünftige Wörter betrachtet oder nur zurückliegende? Woran sieht man das?

Antwort: Ja, es werden auch die zukünftige Wörter berücksichtigt. Das kann man an 'use_causal_mask=False' in unseren MultiHeadAttention Aufruf merken.

2. Wo findet die "Residual connection" statt?

Antwort: Die Residual connections finden in unseren Transformer Block statt. Das sind die Additionen, die als Input für jeden Norm-Layer stattfinden - zwischen den vorherigen Layer Output und der unbearbeiteteten Input.  Z.B. für post_attention_norm_layer werden attention_block_output (Output aus den vorherigen Layer) und composite_embeddings (die unbearbeitete Embeddings) summiert. Analog findet das auch in post_ff_norm_layer statt.
"""

import keras
from keras import ops
from keras import layers
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

### Read in IMDB dataset and pad to maximum length
vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review

(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")

### To Do:  x_train und x_val padden zur Länge, die in Variable maxlen angegeben ist
# Pad the training and validation data to the max token lenght
x_train_padded = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_val_padded = pad_sequences(x_val, padding='post', maxlen=maxlen)

### Alle benötigten Layers (in sinnloser Reihenfolge)
### To Do: sinnvoll benennen

embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
tok_embed_layer = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
pos_embed_layer = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
post_attention_norm_layer = layers.LayerNormalization(epsilon=1e-6)
feedforward_layer = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
post_ff_norm_layer = layers.LayerNormalization(epsilon=1e-6)


### To Do: Layer in der richtigen Reihenfolge anwenden, d.h. alle xxx ersetzen

### Create Embeddings (combine token and positional embedding)
positions = ops.arange(start=0, stop=maxlen, step=1)
tok_embeddings = tok_embed_layer(inputs)
pos_embeddings = pos_embed_layer(positions)
composite_embeddings = tok_embeddings + pos_embeddings

### Transformer block
attention_block_output = attention_layer(composite_embeddings, composite_embeddings, use_causal_mask=False) 
first_norm_output = post_attention_norm_layer(composite_embeddings + attention_block_output)
feedforward_output = feedforward_layer(first_norm_output)
second_norm_output = post_ff_norm_layer(first_norm_output + feedforward_output)

### Classification head

# take the mean across all timesteps
x = layers.GlobalAveragePooling1D()(second_norm_output)
x = layers.Dense(20, activation="relu")(x)
outputs = layers.Dense(2, activation="softmax")(x)

### Compile model
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# model.summary()

# ### To Do:
# # Train model for 2 epochs with a batch size of 32 and x_val, y_val as validation data
history = model.fit(x_train_padded, y_train,
  validation_data=(x_val_padded, y_val),
  epochs=2, batch_size=32
)

# look at the development of the accuracy over the epochs
print("Development of val_accuracy:")
print(history.history['val_accuracy'])

# 1st run: [0.879360020160675, 0.8692399859428406]
# 2nd run: [0.8738800287246704, 0.8525999784469604]
# 3rd run: [0.8777999877929688, 0.8687599897384644]