# RNNs and LSTMs

## Reccurent Neural Networks
### Elman Netwokrs / Simple Reccurent Networks
* A network, where any node feeds back into itself - it's dependent on its own previous output.
* Basis for Long Short-Term Memory (LSTM) networks
### Inference in RNNs
$\mathbf{x}_{t}$ - current input\
$\mathbf{h}_{t}$ - current hidden layer\
$\mathbf{h}_{t-1}$ - previous timestep hidden layer\
$\mathbf{U}$ -weight matrix for **previous hidden layer** output\
$\mathbf{V}$ - weight matrix for **current hidden layer** output\
$\mathbf{W}$ - **current input** weight matrix\
$g()\ and\ f()$ - activation functions

-------

$\mathbf{h}_t = g(\mathbf{Uh}_{t-1} + \mathbf{Wx}_{t})$\
$\mathbf{y}_t = f(\mathbf{Vh}_{t})$

-------

## RNNs as Language Models
* Doesn't have the limited context problem of n-gram models, since the hidden layer can represent information about all preceding words.

### Forward Inference in an RNN language model
$\mathbf{e}_{t}$ - embedding for current word\
$\mathbf{x}_{t}$ - current input\
$\mathbf{h}_{t}$ - current hidden layer\
$\mathbf{h}_{t-1}$ - previous timestep hidden layer\
$\mathbf{E}$ -embedding matrix\
$\mathbf{U}$ -weight matrix for **previous hidden layer** output\
$\mathbf{V}$ - weight matrix for **current hidden layer** output\
$\mathbf{W}$ - **current input** weight matrix\
$g()$ - activation function

-------

$\mathbf{e}_t = \mathbf{Ex}_{t}$\
$\mathbf{h}_t = g(\mathbf{Uh}_{t-1} + \mathbf{We}_{t})$\
$\mathbf{y}_t = softmax(\mathbf{Vh}_{t})$

-------

### Training an RNN language model
#### Self-supervision / self-training
* Take a text and ask the model at each timestep to predict the next word
* No need for labeling (whole text is already there, duh)
* Train to minimize error using cross-entropy

----------------------

$$L_{CE} = - \sum_{w\in V}\mathbf{y}_t[w]\ log\hat{\mathbf{y}_t}[w]$$ 

$$L_{CE}(\hat{\mathbf{y}_t}, \mathbf{y}_t) = -log\hat{\mathbf{y}_t}[w_{t+1}]$$

---------------------
* **teacher forcing** - Always compute the current prediction using the correct (true) sequence, not the predicted tokens

**Question**: How does the Cross-Entropy Loss function work exactly? If y[w] = probability of true word w, what is y_hat[w]? 

### Weight Tying
Since **V** and **E** represent the same information (but with switched axes), we can use: $\mathbf{V} = \mathbf{E}^T$ in the softmax step

## RNNs for other NLP tasks
### Sequence Labeling Tasks
*Such as POS Tagging (assign a label to a token)*

* Same as in the LM case, but output is a distribution over the POS tagset

### Sequence Classification Tasks
*Such as sentiment analysis or spam detection (classifying entire sequences of tokens)*

* **End-to-end Training** - No intermediate loss outputs - only for the final state

or

* **Pooling** - using a mean of all tokens in the seuqence

### Generation with RNN-Based Language Models
* Same as in Bayes:
    1. Sample a word from the \<s> distribution
    2. Use the embedding for the previous word to sample a new one 
    3. Repeat until \</s> is sampled (or limit is reached)

## Stacked and Bidirectional RNN architectures
### Stacked RNNs
* Output from one RNN serves as input in a second one
* Training costs rise quickly with number of stacks

### Bidirectional RNNs
* Use two separate RNNs - one runs left-to-right ($\mathbf{h}^f_t$) and one right-to-left ($\mathbf{h}^b_t$)
$$\mathbf{h}^t = [\mathbf{h}^f_t ; \mathbf{h}^b_t] = \mathbf{h}^f_t\oplus \mathbf{h}^b_t$$
$\oplus$ - vector concatination

**Question**: If $\mathbf{h}^f_t = (x_1, ..., x_t)$ and $\mathbf{h}^b_t = (x_t, ..., x_n)$, wouldn't $\mathbf{h}_t$ include $x_t$ twice after concatination?

## The LSTM
*Long short-term memory*

* Using gates between units to control the flow of information

* Gates consist of a feed-forward layer, followed by a sigmoid function, followed by an element multiplication with the gated layer

#### Forget gate
* Delete information that is no longer needed form the context ($\mathbf{c}_{t-1}$)

----------------------

$\mathbf{f}_t = \sigma(\mathbf{U}_f\mathbf{h}_{t-1} + \mathbf{W}_f\mathbf{x}_{t})$\
$\mathbf{k}_t = \mathbf{c}_{t-1}\odot\mathbf{f}_t $\
$\mathbf{g}_t = tanh(\mathbf{U}_g\mathbf{h}_{t-1} + \mathbf{W}_g\mathbf{x}_{t})$

------------------------

#### Add gate
* Select  information to add to the current context

------------------------

$\mathbf{i}_t = \sigma(\mathbf{U}_i\mathbf{h}_{t-1} + \mathbf{W}_i\mathbf{x}_{t})$\
$\mathbf{j}_t = \mathbf{g}_{t}\odot\mathbf{i}_t $\
$\mathbf{c}_t = \mathbf{j}_t + \mathbf{k}_t$

------------------------

#### Output gate
* Decide what information is required for the current hidden state

--------------------------

$\mathbf{o}_t = \sigma(\mathbf{U}_o\mathbf{h}_{t-1} + \mathbf{W}_o\mathbf{x}_{t})$\
$\mathbf{h}_t = \mathbf{o}_{t}\odot tanh(\mathbf{c}_t) $

--------------------------

**Question**: subscripts of weight matrices **U** and **W** - f,g,i,o - do they correspond to the current gate? I.e. - Are there separate weight matrices for each gate?

### The Encoder-Decoder Model with RNNs
* Input and Output sequences of different lengths - for example Machine Translation

Input -> **Encoder** -> **Contextualized Representation (context vector)** -> **Decoder** -> Output

------------------------

$\mathbf{c} = \mathbf{h}^e_n$\
$\mathbf{h}^d_0 = \mathbf{c}$\
$\mathbf{h}^d_t = g(\^y_{t-1}, \mathbf{h}^d_{t-1},\mathbf{c})$\
$\mathbf{z}_t=f(\mathbf{h}^d_t)$\
$y_t = softmax(\mathbf{z}_t)$

---------------------------

### Training the Encoder-Decoder Model
* end-to-end training
* teacher forcing to speed up training

### Attention

The final hidden state of the encoder acts as a bottleneck - all the information must pass through it

* This means that information from the beginning may be represented worse

#### Attention mechanism
* Allows the decoder to get information fron all hidden states of the encoder




