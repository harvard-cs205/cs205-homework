from P2 import *

import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()


def sort_letters(word):
    return ''.join(sorted(word))

words = sc.textFile('EOWL_words.txt')
anagrams = words.map(lambda word: word.lower().strip().strip('\n')).map(lambda word: (sort_letters(word), [word])).reduceByKey(lambda word1, word2: word1 + word2)

# print anagram with max words
anagrams.sortBy(keyfunc=lambda (anagram, words): len(words), ascending=False).take(1)
