from nltk import FreqDist
import unicodedata

FREQ_THRESHOLD = 2

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
def word_is_capitalized(sent, word):
	return word[0].isupper() and sent[0]['form'] != word

'''
Check if a word is a Ukrainian upper case word and not the first word in the sentence

args:
	sent: the sentence the word is in
	word: the word to check

returns:
	True if the word is a Ukrainian upper case word, False otherwise	
'''
def word_is_capitalized_ukr(sent, word):
    for letter in word:
        if letter.isalpha():
            return unicodedata.name(letter).startswith('CYRILLIC CAPITAL') and sent[0]['form'] != word
    return False

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
def convert_word_to_unk_en(sent, word):
	unk_tag = word
	if word_ends_with(word, 'ing'):
		unk_tag = 'UNK-ing'
	elif word_is_capitalized(sent, word):
		unk_tag = 'UNK-cap'
	elif word_ends_with(word, 'ly'):
		unk_tag = 'UNK-ly'
	elif word_ends_with(word, 'ble'):
		unk_tag = 'UNK-ble'
	elif word_ends_with(word, 'fy'):
		unk_tag = 'UNK-fy'
	elif word_ends_with(word, 'ic'):
		unk_tag = 'UNK-ic'
	elif word_ends_with(word, 'ous'):
		unk_tag = 'UNK-ous'
	elif word_ends_with(word, 'ful'):
		unk_tag = 'UNK-ful'
	elif word_ends_with(word, 'ed'):
		unk_tag = 'UNK-ed'
	elif word_ends_with(word, 'tion'):
		unk_tag = 'UNK-tion'
	return unk_tag

'''
Convert a word to an unknown word tag for French

args:
	sent: the sentence the word is in
	word: the word to convert

returns:
	The unknown word tag for the word
'''
def convert_word_to_unk_fr(sent, word):
	unk_tag = word
	if word_is_capitalized(sent, word):
		unk_tag = 'UNK-cap'
	elif word_ends_with(word, 'ique'):
		unk_tag = 'UNK-ique'
	elif word_ends_with(word, 'iste'):
		unk_tag = 'UNK-iste'
	elif word_ends_with(word, 'elle'):
		unk_tag = 'UNK-elle'
	elif word_ends_with(word, 'eux'):
		unk_tag = 'UNK-eux'
	elif word_ends_with(word, 'er'):
		unk_tag = 'UNK-er'
	elif word_ends_with(word, 'eur'):
		unk_tag = 'UNK-eur'
	elif word_ends_with(word, 'ble'):
		unk_tag = 'UNK-ble'
	elif word_ends_with(word, 'ment'):
		unk_tag = 'UNK-ment'
	elif word_ends_with(word, 'tion'):
		unk_tag = 'UNK-tion'
	return unk_tag

'''
Convert a word to an unknown word tag for Ukrainian

args:
	sent: the sentence the word is in
	word: the word to convert

returns:
	The unknown word tag for the word
'''
def convert_word_to_unk_uk(sent, word):
	unk_tag = word
	if word_is_capitalized_ukr(sent, word):
		unk_tag = 'UNK-cap'
	elif word_ends_with(word, 'іст'):
		unk_tag = 'UNK-іст'
	elif word_ends_with(word, 'ість'):
		unk_tag = 'UNK-ість'
	elif word_ends_with(word, 'ка'):
		unk_tag = 'UNK-ка'
	elif word_ends_with(word, 'ий'):
		unk_tag = 'UNK-ий'
	elif word_ends_with(word, 'ці'):
		unk_tag = 'UNK-ці'
	elif word_ends_with(word, 'ня'):
		unk_tag = 'UNK-ня'
	elif word_ends_with(word, 'овий'):
		unk_tag = 'UNK-овий'
	elif word_ends_with(word, 'ець'):
		unk_tag = 'UNK-ець'
	elif word_ends_with(word, 'ло'):
		unk_tag = 'UNK-ло'
	return unk_tag

'''
Replace the infrequent words in the training set with unknown word tags

args:
	train_sents: the sentences from the training set
	infrequent_words: the infrequent words in the corpus

returns:
	the training set with infrequent words replaced with unknown word tags
'''
def replace_infrequent_words_train(train_sents, infrequent_words, lang):
	for sent in train_sents:
		for token in sent:
			word = token['form']
			# Replace the word with an unknown word tag if it is infrequent
			if word in infrequent_words:
				match lang:
					case 'en':
						token['form'] = convert_word_to_unk_en(sent, word)
					case 'fr':
						token['form'] = convert_word_to_unk_fr(sent, word)
					case 'uk':
						token['form'] = convert_word_to_unk_uk(sent, word)
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
def replace_infrequent_words_test(test_sents, train_sents, infrequent_words, lang):
	train_set_words = get_train_set_words(train_sents)
	for sent in test_sents:
		for token in sent:
			word = token['form']
			# Replace the word with an unknown word tag if it is infrequent or not in the training set
			if word in infrequent_words or word not in train_set_words:
				match lang:
					case 'en':
						token['form'] = convert_word_to_unk_en(sent, word)
					case 'fr':
						token['form'] = convert_word_to_unk_fr(sent, word)
					case 'uk':
						token['form'] = convert_word_to_unk_uk(sent, word)
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
