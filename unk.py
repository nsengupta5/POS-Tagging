from nltk import FreqDist
import unicodedata

FREQ_THRESHOLD = 2
en_suffixes = ['ing', 'ly', 'ble', 'fy', 'ic', 'ous', 'ful', 'ed', 'tion']
fr_suffixes = ['ique', 'iste', 'elle', 'eux', 'er', 'eur', 'ble', 'ment', 'tion']
uk_suffixes = ['іст', 'ість', 'ка', 'ий', 'ці', 'ня', 'овий', 'ець', 'ло']

'''
Check if a word ends with a suffix

args:
	word: the word to check
	suffix: the suffix to check

returns:
	True if the word ends with the suffix, False otherwise
'''
def word_ends_with(word, suffix):
	suff_len = len(suffix)
	return word[-suff_len:] == suffix

'''
Check if a word is capitalized and not the first word in the sentence

args:
	sent: the sentence the word is in
	word: the word to check

returns:
	True if the word is capitalized and not the first word in the sentence, False otherwise
'''
def word_is_capitalized(sent, word, is_uk):
	# Check if the language is Ukrainian
	if is_uk:
		for letter in word:
			if letter.isalpha():
				return unicodedata.name(letter).startswith('CYRILLIC CAPITAL') and sent[0]['form'] != word
	else:
		return word[0].isupper() and sent[0]['form'] != word

'''
Find the infrequent words in the corpus

args:
	sents: the sentences from the corpus

returns:
	A list of infrequent words
'''
def find_infrequent_words(sents):
	# Initialize the infrequent words
	infrequent_words = []

	# Get the frequency of each word in the corpus
	word_freq = FreqDist([token['form'] for sent in sents for token in sent])

	# Add each infrequent word to the list
	for word in word_freq:
		if word_freq[word] < FREQ_THRESHOLD:
			infrequent_words.append(word)

	return infrequent_words

'''
Convert a word to an unknown word tag for English

args:
	sent: the sentence the word is in
	word: the word to convert

returns:
	The unknown word tag for the word
'''
def convert_word_to_unk(sent, word, suffixes, is_uk, catch_all):
	# Replace the word with an general unknown word tag if catch_all is True
	if catch_all:
		unk_tag = 'UNK'
	else:
		unk_tag = word

	# Replace the word with the capitalized unknown word tag if it is capitalized and not the first word in the sentence
	if word_is_capitalized(sent, word, is_uk):
		unk_tag = 'UNK-cap'
	else:
		# Replace the word with the unknown word tag for the suffix if it has a known suffix
		for suffix in suffixes:
			if word_ends_with(word, suffix):
				unk_tag = 'UNK-' + suffix
				break

	return unk_tag

'''
Replace the infrequent words in the training set with unknown word tags

args:
	train_sents: the sentences from the training set
	infrequent_words: the infrequent words in the corpus

returns:
	the training set with infrequent words replaced with unknown word tags
'''
def replace_infrequent_words_train(train_sents, infrequent_words, lang, catch_all):
	for sent in train_sents:
		for token in sent:
			word = token['form']
			# Replace the word with an unknown word tag if it is infrequent
			if word in infrequent_words:
				match lang:
					case 'en':
						token['form'] = convert_word_to_unk(sent, word, en_suffixes, False, catch_all)
					case 'fr':
						token['form'] = convert_word_to_unk(sent, word, fr_suffixes, False, catch_all)
					case 'uk':
						token['form'] = convert_word_to_unk(sent, word, uk_suffixes, True, catch_all)
	return train_sents

'''
Replace the infrequent words in the test set with unknown word tags

args:
	test_sents: the sentences from the test set
	train_sents: the sentences from the training set
	infrequent_words: the infrequent words in the training set

returns:
	the test set with infrequent words replaced with unknown word tags
'''
def replace_infrequent_words_test(test_sents, train_sents, infrequent_words, lang, catch_all):
	train_set_words = get_train_set_words(train_sents)
	for sent in test_sents:
		for token in sent:
			word = token['form']
			# Replace the word with an unknown word tag if it is infrequent or not in the training set
			if word in infrequent_words or word not in train_set_words:
				match lang:
					case 'en':
						token['form'] = convert_word_to_unk(sent, word, en_suffixes, False, catch_all)
					case 'fr':
						token['form'] = convert_word_to_unk(sent, word, fr_suffixes, False, catch_all)
					case 'uk':
						token['form'] = convert_word_to_unk(sent, word, uk_suffixes, True, catch_all)
	return test_sents

'''
Get the words from the training set

args:
	train_set: the sentences from the training set

returns:
	A list of words from the training set
'''
def get_train_set_words(train_set):
	train_set_words = []
	for sent in train_set:
		for token in sent:
			train_set_words.append(token['form'])
	return train_set_words
