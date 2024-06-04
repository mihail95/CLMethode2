###################
##### Aufgabe #####
###################
# Gegeben folgendes Korpus sollen für k = 10 für jeden Lernschritt von BPE
# a) das Vokabular und
# b) die erlernten Merge Regeln ausgegeben werden.


def add_symbols(corpus, vocab):
    for tokenized_string in corpus.keys():
        symbols = tokenized_string.split(" ")
        for symbol in symbols:
            vocab.add(symbol)
    return vocab

def count_bigrams(corpus):
    bigrams = {}
    concatinatedCorpus = "".join(corpus.keys()).replace(" ","")
    for idx in range(len(concatinatedCorpus)):
        bigram = concatinatedCorpus[idx:idx+2]
        if len(bigram) == 2 and bigram[0] != "_":
            if bigram not in bigrams:
                bigrams[bigram] = concatinatedCorpus.count(bigram)
     
    return bigrams

# Korpus (C) mit absoluten Frequenzen
corpus = {
    "l o w _": 5,
    "l o w e r _": 2,
    "n e w e r _": 6,
    "w i d e r _": 3,
    "n e w _": 2,
}

# Anzahl gewünschter Merge Regeln
k = 1
# Merge Regeln
merges = []

# Vokabular (V)
vocab = set()
vocab = add_symbols(corpus, vocab)
print(f"Merge Regeln (k=0): {merges}")
print(f"Vokabular (k=0): {vocab}")
print()

# Lernen der Regeln und Ausgabe
for i in range(k):
    # Bigrams Dict
    bigrams = count_bigrams(corpus)
    print(bigrams)
    # Find most frequent pair of tokens
    mostFreqBigram = max(bigrams, key= bigrams.get)
    print(mostFreqBigram)

    # Concatinate
    # Update rules
    # ...
    # print(f"Merge Regeln (k={i+1}): {merges}")
    # print(f"Vokabular (k={i+1}): {vocab}")
    # print()
    ...