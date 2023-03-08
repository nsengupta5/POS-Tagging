from nltk import FreqDist, WittenBellProbDist

START = '<s>'
BINS = 1e5
END = '</s>'

def get_smoothed(mappings):
	smoothed = {}
	for mapping in mappings:
		smoothed[mapping] = WittenBellProbDist(FreqDist(mappings[mapping]), bins=BINS)
	return smoothed

def get_transmission_prob_matrix(tags, train_sents):
	# Initialize the transition matrix
	transitions = {t: [] for t in tags}
	transitions[START] = []

	# Add the first word of each sentence to the START tag
	for sent in train_sents:
		transitions[START].append(sent[0]['upos'])

		# Add the transition from the previous word to the current word
		for index, token in enumerate(sent[1:]):
			transitions[sent[index - 1]['upos']].append(token['upos'])

		# Add the transition from the last word to the END tag
		transitions[sent[-1]['upos']].append(END)

	# Smooth the transition matrix
	return get_smoothed(transitions)

def get_emission_prob_matrix(train_sents, tags):
	# Initialize the emission matrix
	emissions = {t: [] for t in tags}

	# Add the word to the tag
	for sent in train_sents:
		for token in sent:
			emissions[token['upos']].append(token['form'])

	# Smooth the emission matrix
	return get_smoothed(emissions)

