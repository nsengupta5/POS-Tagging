from p1 import find_accuracy, TREEBANK
from unk import en_suffixes, fr_suffixes, uk_suffixes
import matplotlib.pyplot as plt

test_num = 1
en_suffixes_bak = en_suffixes.copy()
fr_suffixes_bak = fr_suffixes.copy()
uk_suffixes_bak = uk_suffixes.copy()

'''
Prints the title of a test

args:
	test_title: the title of the test
'''
def print_test_title(test_title):
	print(test_title)
	print('=' * len(test_title))

'''
Runs a test

args:
	test: the test to run
'''
def run_test(test):
	global test_num
	test(test_num)
	test_num += 1
	print()

'''
Sets the suffixes for a given language

args:
	suffixes: the suffixes to set
	new_suffixes: the new suffixes to set
'''
def set_suffixes(suffixes, new_suffixes):
	suffixes.clear()
	suffixes.extend(new_suffixes)

'''
Sets up the experiment
'''
def setup():
	# Add the treebanks to the dictionary
	TREEBANK['en'] = 'UD_English-GUM/en_gum'
	TREEBANK['fr'] = 'UD_French-Rhapsodie/fr_rhapsodie'
	TREEBANK['uk'] = 'UD_Ukrainian-IU/uk_iu'

'''
Tests the accuracy with the English suffix 'ing' and capitalization

args:
	test_num: the number of the test
'''
def test_ing_and_cap_en(test_num):
	print_test_title(f"TEST {test_num}: Testing 'ing' and cap for English")
	sub = ['ing']
	set_suffixes(en_suffixes, sub)
	find_accuracy('en', True)
	# Reset suffixes
	set_suffixes(en_suffixes, en_suffixes_bak)

'''
Tests the accuracy with all suffixes, capitalization, and catch all for all languages

args:
	test_num: the number of the test
'''
def test_all_tags_with_catch_all(test_num):
	print_test_title(f"TEST {test_num}: Testing all tags with catch all")
	find_accuracy('en', True, True)
	find_accuracy('fr', True, True)
	find_accuracy('uk', True, True)

'''
Tests the accuracy with only the catch all for all languages
'''
def test_catch_all(test_num):
	print_test_title(f"TEST {test_num}: Testing only catch all")
	# No suffixes
	set_suffixes(en_suffixes, [])
	set_suffixes(fr_suffixes, [])
	set_suffixes(uk_suffixes, [])
	find_accuracy('en', True, True)
	find_accuracy('fr', True, True)
	find_accuracy('uk', True, True)
	# Reset suffixes
	set_suffixes(en_suffixes, en_suffixes_bak)
	set_suffixes(fr_suffixes, fr_suffixes_bak)
	set_suffixes(uk_suffixes, uk_suffixes_bak)

'''
Generates the suffix efficiency graphs for each language
'''
def generate_suffix_graph():
	print_test_title("Generating suffix efficiency graphs for each language")

	# Get the raw accuracy for each language
	en_raw_accuracy = find_accuracy('en', False)
	fr_raw_accuracy = find_accuracy('fr', False)
	uk_raw_accuracy = find_accuracy('uk', False)

	suffix_mapping = {
		'fr': ["French", fr_suffixes, fr_raw_accuracy, fr_suffixes_bak],
		'en': ["English", en_suffixes, en_raw_accuracy, en_suffixes_bak],
		'uk': ["Ukrainian", uk_suffixes, uk_raw_accuracy, uk_suffixes_bak]
	}

	for lang, [long_lang, suffixes, raw_accuracy, suffixes_bak] in suffix_mapping.items():
		suffix_efficiency = []
		# Get the accuracy for each suffix
		for suffix in suffixes:
			set_suffixes(suffixes, [suffix])
			accuracy = find_accuracy(lang, True)
			suffix_efficiency.append((suffix, accuracy - raw_accuracy))
			set_suffixes(suffixes, suffixes_bak)

		# Sort the suffixes by efficiency
		suffix_efficiency.sort(key=lambda x: x[1], reverse=True)
		suffixes, efficiency = zip(*suffix_efficiency)

		# Plot the suffix efficiency graph
		plt.bar(suffixes, efficiency)
		plt.title(f"Suffix Efficiency for {long_lang}")
		plt.xlabel("Suffix")
		plt.ylabel("Accuracy Gain (%)")
		plt.savefig(f"./plots/{lang}_suffix_efficiency.png")
		plt.close()

def main():
	setup()
	run_test(test_ing_and_cap_en)
	run_test(test_catch_all)
	run_test(test_all_tags_with_catch_all)
	generate_suffix_graph()

if __name__ == "__main__":
	main()
