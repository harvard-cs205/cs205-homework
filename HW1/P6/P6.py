import findspark
findspark.init()
import pyspark
import re

sc = pyspark.SparkContext(appName="P6")
sc.setLogLevel('ERROR')

is_invalid = r"(?:\A[0-9]+\Z)|(?:\A[A-Z]+\Z)|(?:\A[A-Z]+\.\Z)"

#read the file, and split based on spaces
shakespeare = sc.textFile("Shakespeare.txt")
words = shakespeare.flatMap(lambda x: x.split())

#filter out words which are invalid based on our regex
words = words.filter(lambda x: not (re.match(is_invalid, x) != None)).collect()

#convert the words into a set of 3tuples
tuples = sc.parallelize([((words[i], words[i+1], words[i+2]), 1) for i in xrange(len(words)-2)])

def combine(count1, count2):
	return count1+count2

tuples_counts = tuples.reduceByKey(combine)

tuples_counts = tuples_counts.map(lambda ((w1,w2,w3),count): ((w1,w2),[(w3,count)]))

#tuples_counts_list = tuples_counts.groupByKey()
tuples_counts_list = tuples_counts.reduceByKey(lambda list1, list2: list1+list2)

nowis = tuples_counts_list.filter(lambda ((w1,w2),v): w1=="Now" and w2=="is").collect()

#for the Spark 1.5.0 bug
tuples_counts_list = tuples_counts_list.map(lambda x:x).cache()

import numpy as np

#words_weights is a list of (word, weight) tuples
def sample_randomly(words_weights):	
	total_word_count = float(sum([x[1] for x in words_weights]))
	probabilities = [x[1]/total_word_count for x in words_weights]

	#from http://stackoverflow.com/questions/6432499/how-to-do-weighted-random-sample-of-categories-in-python
	word_index = np.array(probabilities).cumsum().searchsorted(np.random.rand())
	selected_word = words_weights[word_index][0]

	return selected_word

def get_next_word(word1, word2, tuple_counts):
	next_word_counts = tuple_counts.lookup((word1, word2))[0]

	if next_word_counts:
		next_word = sample_randomly(next_word_counts)

		return next_word
	else:
		print "Words not found:", word1, word2
		return "----"

def get_random_words(tuple_counts):
	return tuple_counts.takeSample(True,1)[0]

def build_sentence(tuple_counts, length=20, word1=None, word2=None):
	if word1==None or word2==None:
		((word1, word2), next_words) = get_random_words(tuples_counts_list)

	sentence = [word1, word2]

	while len(sentence) < length:
		current_first_word, current_second_word = sentence[-2], sentence[-1]
		next_word = get_next_word(current_first_word, current_second_word, tuple_counts)
		sentence.append(next_word)			

	return " ".join(sentence).replace("\n", " ").replace("\r", " ")


print build_sentence(tuples_counts_list)
print build_sentence(tuples_counts_list)
print build_sentence(tuples_counts_list)
print build_sentence(tuples_counts_list, word1="Now", word2="is")
print build_sentence(tuples_counts_list, word1="Now", word2="is")
print build_sentence(tuples_counts_list, word1="Now", word2="is")
print build_sentence(tuples_counts_list, word1="Now", word2="is")
print build_sentence(tuples_counts_list, word1="Now", word2="is")
print build_sentence(tuples_counts_list, word1="We", word2="will")
print build_sentence(tuples_counts_list, word1="You", word2="are")

