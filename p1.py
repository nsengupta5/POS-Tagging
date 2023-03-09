from matrix import get_transmission_prob_matrix, get_emission_prob_matrix
from unk import find_infrequent_words, replace_infrequent_words_train, replace_infrequent_words_test
from conllu import parse_incr
from io import open
from math import log
from sys import argv

TREEBANK = {}
START = '<s>'
END = '</s>'

'''
Returns a sequence of the most probable tags for a given sentence using the Viterbi algorithm

args:
	tags: a list of tags
	emission_probs: a dictionary of dictionaries of smoothed emission probabilities
	transition_probs: a dictionary of dictionaries of smoothed transition probabilities
	test_sentence: a list of tokens from the test corpus

returns:
	A sequence of the most probable tags for the given sentence
'''
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

'''
Gets the accuracy of the Viterbi algorithm for the test corpus

args:
	tags: a list of tags
	emission_matrix: a dictionary of dictionaries of smoothed emission probabilities
	transition_matrix: a dictionary of dictionaries of smoothed transition probabilities
	test_sents: a list of sentences from the test corpus

returns:
	The accuracy of the Viterbi algorithm for the test corpus
'''
def get_accuracy(tags, emission_matrix, transition_matrix, test_sents):
	correct = 0
	total = 0

	for sent in test_sents:
		predicted_tags = viterbi_algorithm(tags, emission_matrix, transition_matrix, sent)
		# Get the accuracy for each word in the test sentence based on the predicted tag
		for token_index, token in enumerate(sent):
			if token['upos'] == predicted_tags[token_index]:
				correct += 1
			total += 1
	return correct / total

'''
Gets the path to the training corpus for the given language

args:
	lang: the language of the corpus

returns:
	The path to the training corpus
'''
def train_corpus(lang): 
	return TREEBANK[lang] + '-ud-train.conllu'

'''
Gets the path to the test corpus for the given language

args:
	lang: the language of the corpus

returns:
	The path to the test corpus
'''
def test_corpus(lang):
	return TREEBANK[lang] + '-ud-test.conllu'

'''
Removes contractions from the corpus

args:
	sent: a sentence from the corpus

returns:
	A sentence with contractions removed
'''
def prune_sentence(sent):
	return [token for token in sent if type(token['id']) is int]

'''
Gets the sentences from the corpus

args:
	path: the path to the corpus

returns:
	A list of sentences from the corpus
'''
def conllu_corpus(path):
	data_file = open(path, 'r', encoding='utf-8')
	sents = list(parse_incr(data_file))
	return [prune_sentence(sent) for sent in sents]

'''
Runs the experiment for the given language and unk option

args:
	lang: the language of the corpus
	use_unk: whether or not to use <unk> for infrequent words
'''
def run_experiment(lang, use_unk):
	train_sents = conllu_corpus(train_corpus(lang))
	test_sents = conllu_corpus(test_corpus(lang))
	tags = set([token['upos'] for sent in train_sents for token in sent])

	if use_unk:
		# Replace infrequent words with <unk>
		infrequent_words = find_infrequent_words(train_sents)

		train_sents = replace_infrequent_words_train(train_sents, infrequent_words, lang)
		test_sents = replace_infrequent_words_test(test_sents, train_sents, infrequent_words, lang)

	# Get the transition and emission matrices
	transition_matrix = get_transmission_prob_matrix(train_sents)
	emission_matrix = get_emission_prob_matrix(train_sents)

	# Get the accuracy of the model using the Viertbi algorithm
	accuracy = get_accuracy(tags, emission_matrix, transition_matrix, test_sents)

	# Print the accuracy
	match lang:
		case 'en':
			print('English Accuracy: {:.2%}'.format(accuracy))
		case 'fr':
			print('French Accuracy: {:.2%}'.format(accuracy))
		case 'uk':
			print('Ukrainian Accuracy: {:.2%}'.format(accuracy))
		case _:
			print('Accuracy: {:.2%}'.format(accuracy))

'''
Throws an error if the arguments are invalid
'''
def throw_err():
	print('ERROR: Invalid argument USAGE: python3 viterbi.py [en|fr|uk|all] [unk]')
	exit()

'''
Checks if the given argument for the language is valid

args:
	arg: the argument for the language
	run_all: whether or not to run the experiment for all languages
	lang: the language of the corpus

returns:
	run_all: whether or not to run the experiment for all languages
	lang: the language of the corpus
'''
def check_valid_lang(arg, run_all, lang):
	languages = ['en', 'fr', 'uk']
	# Check if to run all languages
	if arg == 'all':
		run_all = True
	# Check if the language is valid
	if arg in languages:
		lang = argv[1]
	else:
		throw_err()
	return run_all, lang

def main():
	# Add the treebanks to the dictionary
	TREEBANK['en'] = 'UD_English-GUM/en_gum'
	TREEBANK['fr'] = 'UD_French-Rhapsodie/fr_rhapsodie'
	TREEBANK['uk'] = 'UD_Ukrainian-IU/uk_iu'

	lang = ''
	run_all = False
	use_unk = False

	# Check the arguments
	match len(argv):
		# No arguments (Default to English)
		case 1:
			lang = 'en'
		# One argument (Language)
		case 2:
			run_all, lang = check_valid_lang(argv[1], run_all, lang)
		# Two arguments (Language and unk)
		case 3:
			run_all, lang = check_valid_lang(argv[1], run_all, lang)
			if argv[2] == 'unk':
				use_unk = True
			else:
				throw_err()
		# Invalid number of arguments
		case _:
			throw_err()

	if run_all:
		run_experiment('en', use_unk)
		run_experiment('fr', use_unk)
		run_experiment('uk', use_unk)
	else:
		run_experiment(lang, use_unk)
	
if __name__ == '__main__':
	main()
