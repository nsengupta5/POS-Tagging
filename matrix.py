from nltk import FreqDist, WittenBellProbDist

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
	smoothed = {}
	for mapping in mappings:
		# Smooth the mapping using Witten-Bell smoothing
		smoothed[mapping] = WittenBellProbDist(FreqDist(mappings[mapping]), bins=BINS)
	return smoothed

'''
Gets the transition probability matrix

args:
	tags: a list of tags
	train_sents: a list of sentences from the training corpus

returns:
	A smoothed transition probability matrix
'''
def get_transmission_prob_matrix(tags, train_sents):
	# Initialize the transition matrix
	transitions = {t: [] for t in tags}
	transitions[START] = []

	for sent in train_sents:
		# Add the first word of each sentence to the START tag
		transitions[START].append(sent[0]['upos'])

		# Add the transition from the previous word to the current word
		for index, token in enumerate(sent[1:]):
			transitions[sent[index - 1]['upos']].append(token['upos'])

		# Add the transition from the last word to the END tag
		transitions[sent[-1]['upos']].append(END)

	# Smooth the transition matrix
	return get_smoothed(transitions)

'''
Gets the emission probability matrix

args: 
	tags: a list of tags
	trains_sents: a list of sentences from the training corpus

returns:
	A smoothed emission probability matrix
'''
def get_emission_prob_matrix(tags, train_sents):
	# Initialize the emission matrix
	emissions = {t: [] for t in tags}

	# Add the word to the tag
	for sent in train_sents:
		for token in sent:
			emissions[token['upos']].append(token['form'])

	# Smooth the emission matrix
	return get_smoothed(emissions)
