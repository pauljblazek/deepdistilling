# Deep distilling

This repository includes code to produce results in the paper "Automated discovery of algorithms from data", which can be read at https://rdcu.be/dy2Go. The preprint is at https://arxiv.org/abs/2111.08275.

This paper seeks to fully demonstrate the mechanistic interpretability of ENNs and show that it can be used for discovery of novel algorithms. We have developed deep distilling to automatically condense the weights in an ENN into functioning computer code. This mechanistic interpretability allows for more readily discernible explainability, both by condensing a large number of parameters into a small number of lines of code, as well as by interpreting many aspects of the weights into such forms as for-loops, distilled variables, and logical functions.

Our original work on essence neural networks (ENNs) can be accessed at https://www.nature.com/articles/s43588-021-00132-w or with this shareable PDF link: https://rdcu.be/cyfGB.

## Getting Started

Each of the two learning tasks shown here have a corresponding "distill_*.py" file that sets hyperparameters, imports the ENN weights from corresponding .csv files, condense the ENN to code, and then tests that new code on several test sets. They also have a corresponding "output_*.py" file where the distilled code is written.

### Prerequisites

This code was implemented in Python 3.11.5

### Results

Run the cell in the examples.ipynb notebook. Code is distilled on the shape orientation and MAXSAT problems. Then it is tested against various test sets as described in the paper.

## Files

### Results files

- examples.ipynb : notebook where code can be called to run the ENN condenser for the MAXSAT and orientation problems

- output_*.py : files to which deep distilling writes its output code for the given problem

### Deep distilling files (in /src/)

- distill.py : general code used to call the necessary functions to condense an ENN and then test it; calls into load_dd.py

- load_dd.py : general code used to organize the ENN condensing algorithm, calling the below functions as needed

- cluster.py : file containing the Cluster class, which is used to store groups of neurons and their connectivity patterns

- clustering.py : file containing the algorithm for grouping neurons based on related connectivity patterns. It proceeds by a long series of checks against various types of patterns and then determines for a group what kind of for-loop is best suited for iterating over it.

- forloop.py : file containing the algorithm for producing the code strings. It does this by writing appropriate for-loops and then within them translating the connectivity patterns found in Clusters into Python-style code

- neuron.py : file containing the Neuron class, which contains the weights and connectivity pattern of a neuron

- utils.py : contains some ancillary functions needed, such as rescaling weights and organizing indices of incoming neurons, as well as many short functions used in discovering for-loops and groups

### Problem-specific files (in /src/)

- distill_*.py : file to run distilling for the given problem

- general_maxsat.py : file containing pre-distilled code that generalizes the maxsat problem to any size (i.e., any number of clauses and variables)

- maxsat.py : file containing the code necessary to generate test data for maxsat and test it against the distilled code

### Data files (in /data/)

Within the data/orientation and data/sat folders, ENN weights and biases are stored in the ENN subfolders, and training and test sets are stored in the train and test subfolders.