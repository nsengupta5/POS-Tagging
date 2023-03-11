from math import log
from matrix import START, END

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
