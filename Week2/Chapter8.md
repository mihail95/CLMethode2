# Sequence Labeling for POS and Named Entities

## Hidden Markov Models
### Markov Chains
* Probabilities of sequences of random variables
* Markov Assumption - only the current state matters when predicting the future
$$P(q_i=a|q_1...q_{i-1}) = P(q_i=a|q_{i-1})$$

#### Specification:
Set of N states: $Q = q_1q_2...q_N$\
Transition probability matrix (from i to j in $a_{ij}$): $A=a_{11}a_{12}...a_{N1}...a_{NN}$\
Initial probability distribution: $\pi = \pi_1,\pi_2,...,\pi_N$

#### Figure 8.8a:
8.4 - hot hot hot hot = 0.1 * 0.6 * 0.6 * 0.6 = 0.0216\
8.5 - cold hot cold hot = 0.7 * 0.1 * 0.1 * 0.1 = 0.0007\

### Hidden Markov Model

#### Specification:
$Q, A, \pi$ - same as the Markov Chains\
Observation likelihoods (emission probabilities): $B = b_i(o_t)$

* Markov Assumption
* Output Independence - The observation probability $o_i$ depends only on the state $q_i$

#### Decoding
Given an HMM $\lambda = (A,B)$ and a sequence of observations $O = o_1, o_2, ...., o_T$, find the most probable sequence of states $Q = q_1q_2q_3...q_T$

### The Viterbi Algorithm
1. Initialize
    1. Set the 1st column depending on the initial prob distribution and observation likelihoods\
    $\pi_s * b_s(o_1)$
    2. Set all backpointers in column 1 to 0
2. Recursion
    1. Find the maximum viterbi probability $vit[s',t-1] * a_{s',s} * b_s(o_t)$, where:\
    $vit[s',t-1]$ is the **previous Voterbi path probability**,\
    $a_{s',s}$ is the **transition probability** from the previous to the current state, and\
    $b_s(o_t)$ is the **state observation likelihood** of the observation symbol, given the current state
    2. Set the backpointer to node that gave the maximum
3. Calculate best path backwards

## Conditional Random Fields


