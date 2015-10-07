import numpy as np
import matplotlib.pyplot as plt
import re

import findspark
findspark.init()
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName('P4').setMaster('local')
sc = SparkContext(conf=conf)

# Implemnts a graph for a Markov chain based geenration of phrases based on word occurences

def counter(l):
	b = []
	for i in range(len(l)):
		a = (l[i], l.count(l[i]))
		if a not in b:
			b.append(a)
	return b

def getGraph(source):
	lines = sc.textFile(source)
	
	words = lines.flatMap(lambda line: line.split())
	words = words.filter(lambda word: word.isupper() == False)
	words = words.filter(lambda word: word.isdigit() == False)
	words = words.filter(lambda word: word[0:-1].isupper() == False or word[-1] == '.')

	word_index = words.zipWithIndex()
	index_word = word_index.map(lambda (word, index): (index, word))
	index_plus_one_word = index_word.map(lambda (index, word): (index+1, word))
	index_plus_two_word = index_word.map(lambda (index, word): (index+2, word))

	words_three = index_plus_two_word.join(index_plus_one_word).join(index_word).sortByKey().persist()
	words_three_no_index = words_three.map(lambda (a, ((b, c), d)): ((b, c), (d)))
	words_two_list1 = words_three_no_index.groupByKey()
	words_two_list2 = words_two_list1.map(lambda ((a, b,), c): ((a, b), list(c)))
	words_two_list3 = words_two_list2.map(lambda ((a, b,), c): ((a, b), counter(c)))
	return words_two_list3


graph = getGraph('./shakespeare.txt')


def highest_count(l):
	a = 0
	for i in range(len(l)):
		if l[i][1] > a:
			a = l[i][1]
			b = l[i][0]
	return b

def shakespeare (start):
	Word1 = start[0]
	Word2 = start[1]
	a = [Word1, Word2]
	while len(a) < 20:
		e = graph.lookup((str(Word1), str(Word2)))
		a.append(highest_count(e[0]))
		Word1 = Word2
		Word2 = highest_count(e[0])
	s = ''
	for i in a:
		s = s + i + ' '
	return s

def random_twenty():
	random_two = (graph.takeSample(True, 1)[0])[0]
	return shakespeare(random_two)

print random_twenty()