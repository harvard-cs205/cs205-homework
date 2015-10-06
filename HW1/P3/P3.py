from operator import add

from P2 import *

import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()


def sort_letters(word):
    return ''.join(sorted(word))

words = sc.textFile('EOWL_words.txt')

# set key as string of sorted letters, then reduce by key to make a list
# of all words per sorted combination of letters
anagrams = words.map(lambda word: word.lower().strip().strip('\n')).map(lambda word: (sort_letters(word), [word])).reduceByKey(add)

# print anagram with max words
print anagrams.sortBy(keyfunc=lambda (anagram, words): len(words), ascending=False).take(1)
