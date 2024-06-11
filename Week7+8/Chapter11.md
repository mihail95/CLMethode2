# Fine-Tuning and Masked Language Models

## Bidirectional Transformer Encoders
* Useful for tasks other than autoregressive generation (decoder only models), such like sequence classification and labeling, where the whole context (past and future) could be useful

### Bidirectional models - Architecture
* Same self-attention as casual models
* Only exception: The inputs above the main diagonal are **not** masked

## Training Bidirectional Encoders
* Predicting the next word is trivial, since it's provided in the context
* Use **cloze tasks** instead - fill-in-the-blank

### Learning objective: Masking words
* Masked Language Modeling (MLM)

0. Tokenize
1. Choose a random sample of tokens (15% of the input tokens in BERT)
2. Choose what to do with the token
    1. Replace with a masking token - '[MASK]' - (80% of sample in BERT)
    2. Replace with another random token from the vocabulary (based on unigram probabilities) - (10% of sample in BERT)
    3. Leave it unchanged - (10% of sample in BERT)
3. Include Token and Positional Embeddings
4. Blackbox Bidirectional Transformer Encoder
5. Compute predicion and loss gradient

* Training objective is to predict the original inputs of the masked tokens
* Cross-entropy loss from these predictions guides the training process
* Only the masked tokens are used for the training, all inputsare used in self-attention

### Learning objective: Next Sentence Prediction
* Model is presented with pairs of sentences and has to predict whether they are really adjacent in the training corpus or unrelated. (50/50 in BERT)

0. Tokenize
1. Prepend '[CLS]' to every input pair
2. Place '[SEP]' between the sentences and after the final token.
3. Include Token, Positional and **Segment** (for the first and second segment of the pair) Embeddings
4. Blackbox Bidirectional Transformer Encoder
5. Compute predicion and loss gradient 

* Only the '[CLS]' token is used for the (two-class/binary) prediction 

### Training Regimes
* Multiple languages in separate corpora
* Curse of multilinguality - structures from better represented languages get used in the lesser represented ones

## Contextual Embeddings
* Static embeddings -> represent word *types* (entries in V)
* Contextual embeddings -> represent word instances (word in context)

* Can be used to measure semantic similarity of the same word in different contexts

### Contextual Embeddings and Word Sense
### Word Sense Disambiguation
### Contextual Embeddings and Word Similarity

## Fine Tuning Language Models
* Creating applications on top of pre-trained models by adding application-specific parameters

### Sequence Classification
### Pair-Wise Sequence Classification
### Sequence Labelling

## Span-based Masking
* When the unit of interest is larger than a single word/token

