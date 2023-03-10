from p1 import find_accuracy, TREEBANK
from suffix import *

test_num = 1
en_suffixes_bak = en_suffixes.copy()
fr_suffixes_bak = fr_suffixes.copy()
uk_suffixes_bak = uk_suffixes.copy()

def print_test_title(test_title):
	print(test_title)
	print('=' * len(test_title))

def run_test(test):
	global test_num
	test(test_num)
	test_num += 1
	print()

def set_suffixes(suffixes, new_suffixes):
	suffixes.clear()
	suffixes.extend(new_suffixes)

def setup():
	# Add the treebanks to the dictionary
	TREEBANK['en'] = 'UD_English-GUM/en_gum'
	TREEBANK['fr'] = 'UD_French-Rhapsodie/fr_rhapsodie'
	TREEBANK['uk'] = 'UD_Ukrainian-IU/uk_iu'

def test_ing_and_cap_en(test_num):
	print_test_title(f"TEST {test_num}: Testing 'ing' and cap for English")
	sub = ['ing']
	set_suffixes(en_suffixes, sub)
	find_accuracy('en', True, False)
	set_suffixes(en_suffixes, en_suffixes_bak)

def test_all_tags_with_catch_all_en(test_num):
	print_test_title(f"TEST {test_num}: Testing all tags with catch all for English")
	find_accuracy('en', True, True)

def test_catch_all_en(test_num):
	print_test_title(f"TEST {test_num}: Testing only catch all for English")
	set_suffixes(en_suffixes, [])
	find_accuracy('en', True, True)
	set_suffixes(en_suffixes, en_suffixes_bak)

def test_all_tags_with_catch_all_fr(test_num):
	print_test_title(f"TEST {test_num}: Testing all tags with catch all for French")
	find_accuracy('fr', True, True)

def test_catch_all_fr(test_num):
	print_test_title(f"TEST {test_num}: Testing only catch all for French")
	set_suffixes(fr_suffixes, [])
	find_accuracy('fr', True, True)
	set_suffixes(fr_suffixes, fr_suffixes_bak)

def test_all_tags_with_catch_all_uk(test_num):
	print_test_title(f"TEST {test_num}: Testing all tags with catch all for Ukrainian")
	find_accuracy('uk', True, True)

def test_catch_all_uk(test_num):
	print_test_title(f"TEST {test_num}: Testing only catch all for Ukrainian")
	set_suffixes(uk_suffixes, [])
	find_accuracy('uk', True, True)
	set_suffixes(uk_suffixes, uk_suffixes_bak)

def main():
	setup()
	run_test(test_ing_and_cap_en)
	run_test(test_catch_all_en)
	run_test(test_all_tags_with_catch_all_en)

	run_test(test_catch_all_fr)
	run_test(test_all_tags_with_catch_all_fr)

	run_test(test_catch_all_uk)
	run_test(test_all_tags_with_catch_all_uk)

if __name__ == "__main__":
	main()
