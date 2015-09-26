##############
# Ankit Gupta
# ankitgupta@college.harvard.edu
# CS 205, Problem Set 1, P6
#
##############

from pyspark import SparkContext
sc = SparkContext(appName="Simple")

# Reduce the amount that Spark logs to terminal.
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

import numpy as np 
import itertools
from collections import Counter

# Converts list of words to list of (Word, Count) tuples
# Arguments:
# 	(list[string]) arr: list of strings
# Returns:
# 	(list[(string, int)]) list of tuples (Word, Count) where Count is the number of instances of Word in the input.
def getCounts(arr):
	c = Counter(arr)
	return [(k, v) for k,v in dict(c).iteritems()]

# Generates the next word in the phrase
# Arguments:
# 	(RDD[KV]) words. Key: (Word1, Word2), Value: [(Word3a, Count3a), (Word3b, Count3b), ....]
# 	(string) first: Word1
# 	(string) second: Word2
# Returns:
#	(string) Word3, generated from the Count distribution in the input RDD.
def getNext(words, first, second):
	# Get the choices for the next word
	options = words.lookup((first, second))[0]
	# Extract the words and frequency counts
	next_words = [word for (word, freq) in options]
	frequencies = [freq for (word, freq) in options]
	# Convert the frequencies to probabilities
	total = sum(frequencies)
	probs = [float(freq) / total for freq in frequencies]
	# Draw one from this distribution.
	return np.random.choice(next_words, p=probs)

# Generates a phrase from the word distribution of length num_wanted
# Arguments:
# 	(RDD[KV]) words. Key: (Word1, Word2), Value: [(Word3a, Count3a), (Word3b, Count3b), ....]
# 	(int) num_wanted: length of phrase
# Returns:
# 	(string) Generated phrase.
def generatePhrase(words, num_wanted):
	# Instantiate array that will contain the phrase
	phrase = []
	# Get the first two words, and add them to the array
	start = words.takeSample(True, 1)[0]
	phrase.append(start[0][0])
	phrase.append(start[0][1])
	# Repeatedly get the next word
	for i in range(num_wanted - 2):
		phrase.append(getNext(words, phrase[-2], phrase[-1]))
	# Return the resultant phrase
	return ' '.join(phrase)

# Import the file
lines = sc.textFile('pg100.txt')
lines_with_numbers = lines.zipWithIndex()
# Indicate the lines where the shakespeare actually starts and ends, as opposed to the headers/footers.
start = 172
end = 124368
words = lines_with_numbers\
		.filter(lambda (x, y): y < end and y > start)\
		.keys()\
		.flatMap(lambda x: x.split(' '))\
		.filter(lambda x: x)\
		.filter(lambda x: not x.isdigit())\
		.filter(lambda x: not (x.isalpha() and x.isupper()))\
		.filter(lambda x: not (x[:-1].isupper() and x[-1] == '.'))

# This is a KV RDD with K = word index, V = word
words_with_index = words.zipWithIndex().map(lambda (x,y): (y, x))

# Here we subtract 1 from every index.
offset1 = words_with_index.map(lambda (x, y): (x - 1, y))
# Here we subtract 2 from every index.
offset2 = words_with_index.map(lambda (x, y): (x - 2, y))

# Merge the three, and that gives us phrases that are in tuples ((Word1, Word2), Word3) as we wanted.
# GroupByKey converts this to RDD where each element is of the form ((Word1, Word2), [Word3a, Word3b, Word3c, ...])
# Mapping the values using getCounts gives RDD with elements in the form ((Word1, Word2), [(Word3a, Count3a), (Word3b, Count3b), ...])
joined_words = words_with_index\
				.join(offset1)\
				.join(offset2)\
				.values()\
				.groupByKey()\
				.mapValues(getCounts)\
				.map(lambda x: x)

phrases = []
for i in range(10):
	phrases.append(generatePhrase(joined_words, 20))

print "\n\n\nThe generated phrases are: "
for p in phrases:
	print '\n', p


