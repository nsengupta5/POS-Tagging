from nltk.util import ngrams
from nltk import FreqDist, WittenBellProbDist
from conllu import parse_incr
from io import open
from math import log, exp

LANG = 'en'
TREEBANK = {}
START = '<s>'
END = '</s>'

def get_smoothed(mappings):
	smoothed = {}
	for mapping in mappings.keys():
		smoothed[mapping] = WittenBellProbDist(FreqDist(mappings[mapping]), bins=1e5)
	return smoothed

def get_transmission_prob_matrix(tags, train_sents):
	transitions = {t: [] for t in tags}
	transitions[START] = []

	for sent in train_sents:
		transitions[START].append(sent[0]['upos'])
		for index, token in enumerate(sent[1:]):
			transitions[sent[index - 1]['upos']].append(token['upos'])
		transitions[sent[-1]['upos']].append(END)

	return get_smoothed(transitions)

def get_emission_prob_matrix(train_sents, tags):
	emissions = {t: [] for t in tags}

	for sent in train_sents:
		for token in sent:
			emissions[token['upos']].append(token['form'])

	return get_smoothed(emissions)

def viterbi_algorithm(tags, emission_probs, transition_probs, test_sentence):
	V = [{}]
	backpointers = {}

	for tag in tags:
		V[0][tag] = log(transition_probs[START].prob(tag)) + log(emission_probs[tag].prob(test_sentence[0]['form']))
		backpointers[tag] = [START]

	for word_index in range(1, len(test_sentence)):
		V.append({})
		temp_backpointers = {}
		
		for curr_tag in tags:
			for prev_tag in tags:
				prob = V[word_index - 1][prev_tag] + log(transition_probs[prev_tag].prob(curr_tag)) + log(emission_probs[curr_tag].prob(test_sentence[word_index]['form']))
				if curr_tag not in V[word_index] or prob > V[word_index][curr_tag]:
					V[word_index][curr_tag] = prob
					temp_backpointers[curr_tag] = backpointers[prev_tag] + [curr_tag]
		backpointers = temp_backpointers


	max_prob = 0
	end_tag = None
	for tag in tags:
		prob = V[-1][tag] + log(emission_probs[tag].prob(END))
		if exp(prob) > max_prob:
			max_prob = prob
			end_tag = tag

	return backpointers[end_tag]

def train_corpus(lang): return TREEBANK[lang] + '-ud-train.conllu'

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
	tags = set([token['upos'] for sent in train_sents for token in sent])

	transition_matrix = get_transmission_prob_matrix(tags , train_sents)
	emission_matrix = get_emission_prob_matrix(train_sents, tags)

	print([token['form'] for token in test_sents[1]])
	viterbi_algorithm(tags, emission_matrix, transition_matrix, test_sents[1])

if __name__ == '__main__':
	main()
