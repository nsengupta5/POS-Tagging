from nltk.util import ngrams
from nltk import FreqDist, WittenBellProbDist
from conllu import parse_incr
from io import open

LANG = 'en'
TREEBANK = {}

def get_prob_matrix(mappings, smoothed):
	probs = {}
	for mapping in mappings:
		first_elem, second_elem = mapping
		probabilty = smoothed[first_elem].prob(second_elem)
		probs[mapping] = probabilty
	return probs

def get_smoothed(tags, mappings):
	smoothed = {}
	for tag in set(tags):
		first_elem = [x for (t, x) in mappings if t == tag]
		smoothed[tag] = WittenBellProbDist(FreqDist(first_elem), bins=1e5)
	return smoothed

def get_transmission_prob_matrix(train_sents):
	tags = [token['upos'] for sent in train_sents for token in sent]
	transitions = ngrams(tags, 2)
	smoothed = get_smoothed(tags, transitions)
	transition_probs = get_prob_matrix(transitions, smoothed)
	return transition_probs

def get_emission_prob_matrix(train_sents):
	emissions = [(token['upos'], token['form']) for sent in train_sents for token in sent]
	tags = [tag for tag, _ in emissions]
	smoothed = get_smoothed(tags, emissions)
	emission_probs = get_prob_matrix(emissions, smoothed)
	return emission_probs

def viterbi_algorithm():
	pass

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
	TREEBANK['en'] = 'UD_English-GUM/en_gum'
	TREEBANK['fr'] = 'UD_French-Rhapsodie/fr_rhapsodie'
	TREEBANK['uk'] = 'UD_Ukrainian-IU/uk_iu'

	train_sents = conllu_corpus(train_corpus(LANG))
	test_sents = conllu_corpus(test_corpus(LANG))
	get_transmission_prob_matrix(train_sents)

if __name__ == '__main__':
	main()
