from nltk import FreqDist, WittenBellProbDist, bigrams

START = '<s>'
BINS = 1e5
END = '</s>'

'''
Smooths a matrix using Witten-Bell smoothing

args:
	mappings: a dictionary of dictionaries of lists of words
returns:
	smoothed: a dictionary of dictionaries of smoothed probabilities
'''
def get_smoothed(mappings):
	# Initialize the smoothed matrix
	smoothed = {}
	tags = set([t for (t,_) in mappings])
	for tag in tags:
		words = [w for (t,w) in mappings if t == tag]
		# Smooth the mapping using Witten-Bell smoothing
		smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=BINS)
	return smoothed

'''
Gets the transition probability matrix

args:
	train_sents: a list of sentences from the training corpus

returns:
	A smoothed transition probability matrix
'''
def get_transmission_prob_matrix(train_sents):
	# Initialize the transition matrix
	transitions = []

	for sent in train_sents:
		# Add the first word of each sentence to the START tag
		transitions.append((START, sent[0]['upos']))

		# Add the transition from the previous word to the current word
		for token in bigrams(sent):
			transitions.append((token[0]['upos'], token[1]['upos']))

		# Add the transition from the last word to the END tag
		transitions.append((sent[len(sent) - 1]['upos'], END))

	# Return the smoothed transition matrix
	return get_smoothed(transitions)

'''
Gets the emission probability matrix

args: 
	trains_sents: a list of sentences from the training corpus

returns:
	A smoothed emission probability matrix
'''
def get_emission_prob_matrix(train_sents):
	# Initialize the emission matrix
    emissions = [(token['upos'], token['form']) for sentence in train_sents for token in sentence]

	# Return the smoothed emission matrix
    return get_smoothed(emissions)
