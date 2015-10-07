"""
Written by Jaemin Cheun
Harvard CS205
Assignment 1
October 6, 2015
"""
from random import randint
import numpy as np
import findspark
findspark.init()

from pyspark import SparkContext

# initiaize Spark
sc = SparkContext("local", appName="P6")
sc.setLogLevel("ERROR")
lines = sc.textFile("Shakespeare.txt")

# filtering any words that do not satisfy the conditions in the problem
def wordFilter(word):
	return (not word.isdigit()) and (not word.isupper()) and (not (word.endswith('.') and word[:-1].isupper()))
 
# we make sure we switch the order of index and word, and we cache this 
word_list = lines.flatMap(lambda line: line.split()).filter(wordFilter).zipWithIndex().map(lambda (x,y) : (y,x)).cache()

# get 3 consecutive words, get word1 word2 word3
one_shift = word_list.map(lambda (x,y) : (x-1,y))
two_shift = word_list.map(lambda (x,y) : (x-2,y))

# join the RDDs and make sure the RDD follows the structure given by the homework
consec_words = word_list.join(one_shift).join(two_shift).map(lambda (k, ((w1, w2),w3)): ((w1,w2,w3),1))
consec_words = consec_words.reduceByKey(lambda x,y : x+y)
markov_model = consec_words.map(lambda ((w1,w2,w3),n) : ((w1,w2), (w3,n))).groupByKey().mapValues(lambda l : list(l)).cache()

# use the model to generate text
def generateSentence(model):
	sentence = []
	max_length = 20
	sample = model.takeSample(True, 1)[0] 
	first_word = sample[0][0]
	second_word = sample[0][1]
	sentence += [first_word]
	sentence += [second_word]
	while len(sentence) < max_length:
		# we create an array called bias choice where the possible word3 are repeated according to its value
		bias_choice = []
		third_word_list = model.map(lambda x: x).lookup((first_word, second_word))[0]
		for word, counts in third_word_list:
			for i in xrange(counts):
				bias_choice.append(word)
		# now use random number generator to pick the number: this make sures that the choice is biased by the counts
		k = randint(0,len(bias_choice) -1)
		third_word = bias_choice[k]
		sentence += [third_word]
		first_word = second_word
		second_word = third_word
	sentence=' '.join(sentence)
	return sentence

paragraph = []
for i in xrange(10):
	print(generateSentence(markov_model))









