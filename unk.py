from nltk import FreqDist

FREQ_THRESHOLD = 2

def word_ends_with(word, suffix):
	suff_len = len(suffix)
	return word[-suff_len:] == suffix

def word_starts_with(word, prefix):
	pref_len = len(prefix)
	return word[:pref_len] == prefix

def word_is_capitalized(word):
	return word[0].isupper()

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

def convert_word_to_unk(train_sents, test_sents):
	unk_tag = 'UNK'

def replace_infrequent_words(train_sents, infrequent_words):
	# Replace each infrequent word with the string 'UNK'
	for sent in train_sents:
		for token in sent:
			if token['form'] in infrequent_words:
				token['form'] = 'UNK'
