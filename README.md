# POS Tagging

This project implements a Hidden Markov Model (HMM) for Part-of-Speech (POS) tagging. The model is trained on the English, French, and Ukrainian Universal Dependencies corpuses. It utilizes the Viterbi algorithm to find the most likely sequence of POS tags for a given sentence. It also intruduces complex preprocessing techniques that use observations of an unknown tag to improve the accuracy of the model.

## Dependencies
- `nltk`
- `conllu`
- `matplotlib` (for graph generation)

## Instructions

To execute the script, run the command `python3 p1.py [en|fr|uk|all] [unk]`

`en` will get the accuracy for the English corpus. This is the default corpus if no arguments are provided

`fr` will get the accuracy for the French corpus

`uk` will get the accuracy for the Ukrainian corpus

`all` will get the accuracy for all three corpuses

The `unk` argument if provided will replace infrequent words with the unknown tags

## Experiments

To execute the experiments and generate the suffix graphs, run the command `python3 experiment.py`
