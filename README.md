# Deep distilling

This repository includes the code used to produce the results in the paper "Automated discovery of algorithms from data". We have developed a new method to automatically condense the weights in an essence neural network (ENN) into functioning computer code. Doing so allows for more readily discernible explainability, both by condensing a large number of parameters into a small number of lines of code, as well as by interpreting many aspects of the weights into such forms as for-loops, distilled variables, and logical functions.

## Getting Started

Each of the two learning tasks shown here have a corresponding "distill_*.py" file that sets hyperparameters, imports the ENN weights from corresponding .csv files, condense the ENN to code, and then tests that new code on several test sets. They also have a corresponding "output_*.py" file where the distilled code is written.

### Prerequisites

This code was implemented in Python 3.6.4 with Anaconda 4.4.10, scipy, scikit-learn, Keras 2.2.4, and Tensorflow 2.0.

### Results

Code is distilled or the shape orientation and MAXSAT problems here. Then it is tested against various test sets as described in the paper. The results are written to the screen and stored in the results>output file provided with Code Ocean for reproducible runs.

## Files

### Deep distilling files

- distill.py : general code used to call the necessary functions to condense an ENN and then test it; calls into load_dd.py

- load_dd.py : general code used to organize the ENN condensing algorithm, calling the below functions as needed

- cluster.py : file containing the Cluster class, which is used to store groups of neurons and their connectivity patterns

- clustering.py : file containing the algorithm for grouping neurons based on related connectivity patterns. It proceeds by a long series of checks against various types of patterns and then determines for a group what kind of for-loop is best suited for iterating over it.

- forloop.py : file containing the algorithm for producing the code strings. It does this by writing appropriate for-loops and then within them translating the connectivity patterns found in Clusters into Python-style code

- neuron.py : file containing the Neuron class, which contains the weights and connectivity pattern of a neuron

- utils.py : contains some ancillary functions needed, such as rescaling weights and organizing indices of incoming neurons, as well as many short functions used in discovering for-loops and groups

### Problem-specific files

- distill_*.py : file to run distilling for the given problem

- output_*.py : file to which deep distilling writes its output code for the given problem

- general_maxsat.py : file containing pre-distilled code that generalizes the maxsat problem to any size (i.e., any number of clauses and variables)

- maxsat.py : file containing the code necessary to generate test data for maxsat and test it against the distilled code

### Data files

Within the data>orientation and data>sat folders, ENN weights and biases are stored in the ENN subfolders, and training and test sets are stored in the train and test subfolders.