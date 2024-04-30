"""
Name:        Mihail Chifligarov
Matrikelnr:  108022214940

Aufgabenstellung:

Implementieren Sie den Viterbi-Algorithmus für einen HMM-POS-Tagger.
Orientieren Sie sich dabei am Pseudocode in Abbildung 8.10 in Jurafsky & Martin.

Es seien bereits Emissions- und Uebergangswahrscheinlichkeiten berechnet worden
und abgespeichert in den Binaerdateien emission_probs.pkl und transition_probs.pkl.
Diese Dateien werden als Dictionarys eingelesen und haben folgendes Format:

emissions[(tag,tok)] = prob
transitions[(tag,tag_next)] = prob

Die Trainingssaetze waren gepadded mit <START> und <END>,
diese beiden Spezialwoerter wurden mit dem speziellen Tag "$P" getagged.
Fuer diese Woerter werden Emissionswahrscheinlichkeiten noch wie folgt ergaenzt:
emissions[("$P", "<START>")] = 0.5
emissions[("$P", "<END>")] = 0.5

Für die initiale Wahrscheinlichkeitsverteilung Pi gilt:
Tag "$P" hat eine Wahrscheinlichkeit von 1,
alle anderen Tags haben eine Wahrscheinlichkeit von 0.

Für die Bestimmung der Emissions- und Uebergangswahrscheinlichkeiten  wurde ein Smoothing-Verfahren eingesetzt.
Nehmen Sie fuer unbekannte Woerter und unbekannte Uebergaenge eine Wahrscheinlichkeit von 0.0001 an.

Gegeben diese Wahrscheinlichkeiten, berechnen Sie die wahrscheinlichste Tag-Sequenz fuer die folgenden Saetze:

"<START> Kein Ende der Zeitung . <END>"
"<START> Er hatte keine Lust mehr , jeden Tag mit dem Auto zu fahren . <END>"

Die Ausgabe soll folgendermaßen aussehen:

Sentence: <START> Kein Ende der Zeitung . <END>
<START>	$P
Kein	DET
Ende	N
der	DET
Zeitung	N
.	PUNC
<END>	$P
Probability of best path: 1.1816007236878401e-14

Sentence: <START> Er hatte keine Lust mehr , jeden Tag mit dem Auto zu fahren . <END>
<START>	$P
Er	PRON
hatte	V
keine	DET
Lust	N
mehr	ADV
,	PUNC
jeden	DET
Tag	N
mit	PREP
dem	DET
Auto	N
zu	ADV
fahren	V
.	PUNC
<END>	$P
Probability of best path: 7.72441573033475e-46

Sie dürfen keine externen Libraries außer pickle zum Einlesen der Emissions- und Uebergangswahrscheinlichkeiten benutzen.
Teilen Sie Ihren Code sinnvoll in Funktionen auf und halten Sie sich an die Best Practices zur Übersichtlichkeit und Kommentierung von Programmen.
"""

import pickle
#################################
# Klassen
#################################
class Viterbi:
    """A class for applying the Viterbi algorithm on observations
    
    Initialization Input:
        emmisions:dict: A dictionary of emmision probabilities, structured like this - emissions[(tag,tok)] = prob
        transitions:dict: A dictionary of transition probabilities, structured like this - transitions[(tag,tag_next)] = prob
        states:set: A set of states

    Methods:
        find_best_path(observations:str)
    """
    def __init__(self, emissions:dict, transitions:dict, states:set) -> None:
        self.emissions = emissions
        self.transitions = transitions
        self.states = list(states)
        self.initials = { "$P": 1 }

    def initialize_matrix(self, observationList:list, dimensions:tuple) -> tuple[list, list]:
        """Initializes the viterbi and the backpointer matrices

            Input: 
                observationList:list - A list of observations
                dimensions:tuple - A tuple of matrix dimensions

            Output: viterbiMatrix:2D Array, backpointerMatrix:2D Array
        """

        # Initialize the matrices and fill them with zeroes
        viterbiMatrix = [[0 for t in range(dimensions[1])] for n in range(dimensions[0])]
        backpointerMatrix = [[0 for t in range(dimensions[1])] for n in range(dimensions[0])]

        # Calculate the first row
        for idx,state in enumerate(self.states):
            # Initial probability - 0 if not in the dict
            initialProb = self.initials.get(state, 0)
            # observation/emission probability - 0.0001 if not in the dict
            observationProb = self.emissions.get((state, observationList[0]), 0.0001)
            viterbiMatrix[idx][0] = initialProb * observationProb

        return viterbiMatrix, backpointerMatrix
    
    def get_max_score_in_prev_timestep(self, prevTimestepColumn:list, currentStateIdx:int) -> tuple[float, int]:
        """Gets the maximum score and its index from the previous viterbi timestep
        The score is calculated by multiplying the previous state by its transition probability to the current one

            Input: 
                prevTimestepColumn:list - A list of all state values from the previous timestep
                currentStateIdx:int - The index of the current state

            Output: 
                maxviterbiValue:float - The maximum score
                maxValueNodeIdx:int - The index of the maximum score
        """
        # This 'beautiful' line of code calculates an array of (partial) viterbi scores by multiplying each state from the previous column
        # by the transition probability to the current state. If the transition probability does not exist, it defaults to 0.0001.
        prevViterbiScores = [prevStateScore * self.transitions.get((self.states[prevStateIdx], self.states[currentStateIdx]), 0.0001) for prevStateIdx, prevStateScore in enumerate(prevTimestepColumn)]
        maxviterbiValue = max(prevViterbiScores)
        maxValueNodeIdx = prevViterbiScores.index(maxviterbiValue)

        return maxviterbiValue,maxValueNodeIdx
    
    def recursion_step(self, observationList:list, viterbiMatrix:list, backpointerMatrix:list, dimensions:tuple) -> tuple[list, list]:
        """Itterates through the rest of the matrix and computes the tags and probabilities

            Input:
                observationList:list - A list of observations
                viterbiMatrix:2D Array, backpointerMatrix:2D Array
                dimensions:tuple - A tuple of matrix dimensions

            Output: viterbiMatrix:2D Array, backpointerMatrix:2D Array
        """
        for timeStep in range(1, dimensions[1]):
            for state in range(dimensions[0]):
                # max of prev score * transition
                maxviterbiValue, maxValueNodeIdx = self.get_max_score_in_prev_timestep([row[timeStep-1] for row in viterbiMatrix], state)
                # multiplied by the observation/emission probability - 0.0001 if not in the dict
                observationProb = self.emissions.get((self.states[state], observationList[timeStep]), 0.0001)
                viterbiMatrix[state][timeStep] = maxviterbiValue * observationProb
                backpointerMatrix[state][timeStep] = maxValueNodeIdx

        return viterbiMatrix, backpointerMatrix
    
    def termination_step(self, viterbiMatrix:list, backpointerMatrix:list, dimensions:tuple) -> tuple[list, float]:
        """ Terminates the viterbi algorithm by finding out the best path probability and tracing back the best path to the start

            Inputs: 
                viterbiMatrix:2D Array, backpointerMatrix:2D Array
                dimensions:tuple - A tuple of matrix dimensions
            
            Output:
                bestPath:list - The predicted sequence of tags
                bestPathProb:float - The Probability of the sequence to occur
        """
        lastViterbiColumn = [row[-1] for row in viterbiMatrix]
        bestPathProb = max(lastViterbiColumn)
        bestPathNodeIdx = lastViterbiColumn.index(bestPathProb)
        
        bestPath = [self.states[bestPathNodeIdx]]

        prevPathIdx = bestPathNodeIdx
        for col in range(1, dimensions[1]):
            pathIdx = backpointerMatrix[prevPathIdx][-col]
            bestPath.append(self.states[pathIdx])
            prevPathIdx = pathIdx

        bestPath.reverse()
        return bestPath, bestPathProb

    def find_best_path(self, observations:str) -> tuple[list, float]:
        """ Runs the viterbi algorithm to find the optimal sequence of tags

        Input:
            observations:str: The string of observations, to be tagged

        Output:
            (bestPath, pathProb):
            bestPath:list: The best tag sequence 
            pathProb:float: The probability of the path 
        """
        observationList = observations.split()
        # Set the dimensions of the probability and backpointer matrices
        dimensions = (len(self.states), len(observationList))

        viterbiMatrix, backpointerMatrix = self.initialize_matrix(observationList, dimensions)
        viterbiMatrix, backpointerMatrix = self.recursion_step(observationList, viterbiMatrix, backpointerMatrix, dimensions)        
        bestPath, bestPathProb = self.termination_step(viterbiMatrix, backpointerMatrix, dimensions)

        return bestPath,bestPathProb



#################################
#Funktionen
#################################
def show_matrix(matrix) -> None:
    """Prints a 2D array/matrix in a readable format"""
    for row in matrix:
        print(row)

def extract_states(emissions:dict) -> set:
    """ Extracts a set of all possible States in the emissions dictionary\n
    Input: 
        emissions:dict, structured like emissions[(tag,tok)] = prob
    
    Output:
        stateSet:set, a set of all unique tags from the emmision dictionary      
     """
    
    return { tag[0] for tag in emissions.keys() }

def print_results(observations:str, path:list, probability:float) -> None:
    """ Prints out the results for the current sentence

    Input: 
        observations:str : The string of observations
        path:list : The predicted best path
        probability:float : The probability of that path occuring
    """
    print("----------------------------------------------------------")
    print("Sentence: ", observations, "\n")
    for idx, token in enumerate(observations.split()):
        print(f"{token}\t{path[idx]}")

    print("\nProbability of best path: ", probability)
    print("----------------------------------------------------------\n\n")



        
def run_script(emissions:dict, transitions:dict, sentences_to_tag:list) -> None:

    """
    Funktion, die alle weiteren Funktionen aufruft
    
    """
    stateSet = extract_states(emissions)
    viterbi = Viterbi(emissions, transitions, stateSet)
    
    for observations in sentences_to_tag:
        path, probability = viterbi.find_best_path(observations)
        print_results(observations, path, probability)
        
    
    
################################
# Hauptprogramm
################################

if __name__ == "__main__":
    
    emission_probs_file = "emission_probs.pkl"
    transition_probs_file = "transition_probs.pkl"
    
    # Emissionswahrscheinlichkeiten einlesen
    # Dictionary im Format: emissions[(tag,tok)] = prob
    infile = open(emission_probs_file, "rb")
    emissions = pickle.load(infile)
    infile.close()
    
    # Die Trainingssaetze waren gepadded mit <START> und <END>,
    # diese beiden Spezialwoerter wurden mit dem speziellen Tag "$P" getagged.
    # Fuer diese Woerter muessen Emissionswahrscheinlichkeiten noch wie folgt ergaenzt werden:
    emissions[("$P", "<START>")] = 0.5
    emissions[("$P", "<END>")] = 0.5
    
    
    # Transitionswahrscheinlichkeiten einlesen
    # Dictionary im Format: transitions[(tag,tag_next)] = prob
    infile = open("transition_probs.pkl", "rb")
    transitions = pickle.load(infile)
    infile.close()
        
    sentences_to_tag = ["<START> Kein Ende der Zeitung . <END>",
                       "<START> Er hatte keine Lust mehr , jeden Tag mit dem Auto zu fahren . <END>"
                       ]

    run_script(emissions, transitions, sentences_to_tag)



