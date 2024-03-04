from matrix import get_transmission_prob_matrix, get_emission_prob_matrix
from viterbi import get_accuracy
from unk import find_infrequent_words, replace_infrequent_words_train, replace_infrequent_words_test
from conllu import parse_incr
from io import open
from sys import argv
import logging

TREEBANK = {}

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
Finds the accuracy for the given language and unk option

args:
	lang: the language of the corpus
	use_unk: whether or not to use <unk> for infrequent words
'''
def find_accuracy(lang, use_unk, catch_all=False, use_cap=True):
	train_sents = conllu_corpus(train_corpus(lang))
	test_sents = conllu_corpus(test_corpus(lang))
	tags = set([token['upos'] for sent in train_sents for token in sent])

	if use_unk:
		# Replace infrequent words with <unk>
		infrequent_words = find_infrequent_words(train_sents)

		train_sents = replace_infrequent_words_train(train_sents, infrequent_words, lang, catch_all, use_cap)
		test_sents = replace_infrequent_words_test(test_sents, train_sents, infrequent_words, lang, catch_all, use_cap)

	# Get the transition and emission matrices
	transition_matrix = get_transmission_prob_matrix(train_sents)
	emission_matrix = get_emission_prob_matrix(train_sents)

	# Get the accuracy of the model using the Viertbi algorithm
	accuracy = get_accuracy(tags, emission_matrix, transition_matrix, test_sents)

	unk_str = ' (with <UNK> tags)' if use_unk else ''
	# Print the accuracy
	match lang:
		case 'en':
			print('English Accuracy{}: {:.2%}'.format(unk_str,accuracy))
		case 'fr':
			print('French Accuracy{}: {:.2%}'.format(unk_str,accuracy))
		case 'uk':
			print('Ukrainian Accuracy{}: {:.2%}'.format(unk_str,accuracy))
		case _:
			print('Accuracy{}: {:.2%}'.format(unk_str,accuracy))

	return accuracy

'''
Throws an error if the arguments are invalid
'''
def throw_err():
	fmt = '[%(levelname)s] %(message)s'
	logging.basicConfig(format=fmt, level=logging.DEBUG)
	logging.error('Invalid argument USAGE: python3 viterbi.py [en|fr|uk|all] [unk]')
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
	elif arg in languages:
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

	# Run the experiment
	if run_all:
		find_accuracy('en', use_unk)
		find_accuracy('fr', use_unk)
		find_accuracy('uk', use_unk)
	else:
		find_accuracy(lang, use_unk)
	
if __name__ == '__main__':
	main()
