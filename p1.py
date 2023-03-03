from nltk.util import ngrams
from nltk import FreqDist, WittenBellProbDist
from conllu import parse_incr
from io import open

LANG = 'en'
TREEBANK = {}

def get_tag_counts(tags):
	return FreqDist(tags)

def get_tag_transition_counts(tags):
	return FreqDist(ngrams(tags, 2))

def get_tag_emission_counts(emissions):
	return FreqDist(emissions)

def get_transmission_prob_matrix(train_sents):
	tags = [token['upos'] for sent in train_sents for token in sent]
	tag_counts = get_tag_counts(tags)
	tag_transition_counts = get_tag_transition_counts(tags)
	transition_probs = {}

	for transition, count in tag_transition_counts.items():
		prev_tag, _ = transition
		probabilty = count / tag_counts[prev_tag]
		transition_probs[transition] = probabilty

	return transition_probs

def get_emission_prob_matrix(train_sents):
	emissions = [(token['form'], token['upos']) for sent in train_sents for token in sent]
	tag_counts = get_tag_counts([tag for _, tag in emissions])
	tag_emission_counts = get_tag_emission_counts(emissions)
	emission_probs = {}

	for emission, count in tag_emission_counts.items():
		_, tag = emission
		probabilty = count / tag_counts[tag]
		emission_probs[emission] = probabilty

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
	get_emission_prob_matrix(train_sents)

if __name__ == '__main__':
	main()
