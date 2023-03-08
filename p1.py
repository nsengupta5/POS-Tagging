from nltk import FreqDist, WittenBellProbDist
from conllu import parse_incr
from io import open
from math import log
from sys import argv

LANG = 'en'
TREEBANK = {}
START = '<s>'
BINS = 1e5
END = '</s>'

def get_smoothed(mappings):
	smoothed = {}
	for mapping in mappings.keys():
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

def viterbi_algorithm(tags, emission_probs, transition_probs, test_sentence):
	# Initialize the Viterbi matrix and backpointers
	V = [{}]
	backpointers = {}

	# Initialize the first row of the Viterbi matrix
	for tag in tags:
		V[0][tag] = log(transition_probs[START].prob(tag)) + log(emission_probs[tag].prob(test_sentence[0]['form']))
		backpointers[tag] = [tag]

	# Fill in the rest of the Viterbi matrix
	for word_index in range(1,len(test_sentence)):
		V.append({})
		temp_backpointers = {}
		
		# Find the maximum probability for each tag
		for curr_tag in tags:
			all_probs = []
			for prev_tag in tags:
				prob = V[word_index - 1][prev_tag] + log(transition_probs[prev_tag].prob(curr_tag)) + log(emission_probs[curr_tag].prob(test_sentence[word_index]['form']))
				all_probs.append((prob, prev_tag))
			max_prob, prev_tag = max(all_probs)

			# Add the maximum probability to the Viterbi matrix and the backpointer to the temp backpointers
			V[word_index][curr_tag] = max_prob
			temp_backpointers[curr_tag] = backpointers[prev_tag] + [curr_tag]

		# Update the backpointers
		backpointers = temp_backpointers

	# Find the maximum probability for the END tag
	max_prob, end_tag = max([(V[-1][tag] + log(transition_probs[tag].prob(END)), tag) for tag in tags])

	# Return the most probable sequence of tags
	return backpointers[end_tag]

def get_accuracy(tags, emission_matrix, transition_matrix, test_sents):
	correct = 0
	total = 0

	for sent in test_sents:
		# Get the accuracy for each word in the test sentence
		predicted_tags = viterbi_algorithm(tags, emission_matrix, transition_matrix, sent)
		for token_index, token in enumerate(sent):
			if token['upos'] == predicted_tags[token_index]:
				correct += 1
			total += 1
	return correct / total

def train_corpus(lang): 
	return TREEBANK[lang] + '-ud-train.conllu'

def test_corpus(lang):
	return TREEBANK[lang] + '-ud-test.conllu'

# Remove contractions such as "isn't".
def prune_sentence(sent):
	return [token for token in sent if type(token['id']) is int]

def conllu_corpus(path):
	data_file = open(path, 'r', encoding='utf-8')
	sents = list(parse_incr(data_file))
	return [prune_sentence(sent) for sent in sents]


def main():
	# Add the treebanks to the dictionary
	TREEBANK['en'] = 'UD_English-GUM/en_gum'
	TREEBANK['fr'] = 'UD_French-Rhapsodie/fr_rhapsodie'
	TREEBANK['uk'] = 'UD_Ukrainian-IU/uk_iu'

	# Get the language from the command line (default English)
	if len(argv) > 1:
		LANG = argv[1]
	else:
		LANG = 'en'

	train_sents = conllu_corpus(train_corpus(LANG))
	test_sents = conllu_corpus(test_corpus(LANG))
	tags = set([token['upos'] for sent in train_sents for token in sent])

	# Get the transition and emission matrices
	transition_matrix = get_transmission_prob_matrix(tags , train_sents)
	emission_matrix = get_emission_prob_matrix(train_sents, tags)

	# Get the accuracy of the model using the Viertbi algorithm
	accuracy = get_accuracy(tags, emission_matrix, transition_matrix, test_sents)

	# Print the accuracy
	match LANG:
		case 'en':
			print('English Accuracy: {:.2%}'.format(accuracy))
		case 'fr':
			print('French Accuracy: {:.2%}'.format(accuracy))
		case 'uk':
			print('Ukrainian Accuracy: {:.2%}'.format(accuracy))
		case _:
			print('Accuracy: {:.2%}'.format(accuracy))

if __name__ == '__main__':
	main()