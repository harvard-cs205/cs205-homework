import pyspark
import pickle
from pyspark import SparkContext
sc = SparkContext()

# reading data
def read_data(data):
	return sc.textFile(data)
# sort each word
def word_char(text):
	eowl = read_data(text)
	wordChar = eowl.map(lambda word: sorted(list(word)))
	return wordChar
# group by key to get anagrams
def maximum(textfile):
	keyval = read_data(textfile).map(lambda x: ("".join(sorted(list(x))),x))
	temp = keyval.groupByKey().mapValues(list)
	anagram = temp.map(lambda x: (x[0],(len(x[1]),x[1])))
	maximum = anagram.filter(lambda x: x[1][0] == 11)
	print maximum.collect()

# get the squence of letters in alphabetical order
SortedLetterSequence = word_char('EOWL_words.txt').map(lambda x: "".join(x))

# print the maximum list
maximum('EOWL_words.txt')

