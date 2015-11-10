import csv
import urllib2
import itertools
import math as ma
import pickle
import numpy as np
from random import shuffle
import findspark
import os
import findspark
findspark.init()
import pyspark



def word_cleaner(word):
    
    '''
    INPUT: 
    word - takes in the words of shakespear and returns word if it meets 
    specifications of the problem
    OUTPUT:
    word - only if it meets the specification
    '''
    
    if word == ' ':
        return False

    elif word == word.upper():
        return False

    elif word.isdigit():
        return False
    
    else:
        return True
    
def from_book_to_triples(book):
    
    '''
    
    INPUT: 
    book - takes in a book of data 
    OUTPUT:
    combinations - two word combinations, keyed on the two words, with a list of words that
    come after each of these two word combinations
    
    '''
    book_list_1 = book.collect()[1:]+[""]
    book_list_2 = book_list_1[1:]+[""]
    
    
    book_2 = sc.parallelize(book_list_1)
    book_3 = sc.parallelize(book_list_2)
    
    index_rdd_1 = book.zipWithIndex().map(lambda x: (x[1], x[0]))
    index_rdd_2 = book_2.zipWithIndex().map(lambda x: (x[1], x[0]))
    index_rdd_3 = book_3.zipWithIndex().map(lambda x: (x[1], x[0]))
    
    combinations = index_rdd_1.join(index_rdd_2).join(index_rdd_3)
    triples = combinations.map(lambda (x, y): ((y[0][0], y[0][1], y[1]), 1)).reduceByKey(lambda x, y: x + y)
    reduced_triples = triples.reduceByKey(lambda x, y: x + y)
    indexed_triples = reduced_triples.map(lambda (x, y): ((x[0], x[1]),(x[2], y)))
    list_triples = indexed_triples.groupByKey().mapValues(list)
    
    return list_triples


def word_freq(last_word):

    '''
    Generates how often a word is present
    '''
    freq = []
    for i in last_word:
        freq += [i[0]] * i[1]
    return freq


def shake_markov(triple, sentence_length=10):
    
    '''
    Loops through user defined length and develops sentance based on random markov 
    second order algorithm
    '''
    
    stream = []
    start_words = triple.takeSample(True, 1)
    word_1 = start_words[0][0][0]
    word_2 = start_words[0][0][1]
    last_word = start_words[0][1]
    word_data = (word_1, word_2)
    stream += [word_1] + [word_2]
    while len(stream) < sentence_length:
        freq = word_freq(last_word)
        word_3 = np.random.choice(freq)
        stream += [word_3]
        word_data = (word_data[1], word_3)
        last_word = triple.map(lambda x: x).lookup(word_data)
        last_word = last_word[0]
    
    output = [' '.join(stream)]
    
    print 'The sentance is:', output[0]
    return 'The sentance is:', output[0]
        
        



if __name__ == '__main__':
	sc = pyspark.SparkContext()
	 

	lines = sc.textFile('pg100.txt')
	words = lines.flatMap(lambda line: line.split())
	words = words.filter(word_cleaner)
	out = from_book_to_triples(words)

	sentence_data = []

	for i in range(10):
		sentence_data.append(shake_markov(out, 20))

	pickle.dump(sentence_data, open( "shakespear_sentences.pkl", "wb" ) )











