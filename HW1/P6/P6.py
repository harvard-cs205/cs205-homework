import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
sc = SparkContext()

import random
import re

cap_pattern = re.compile(r'^[A-Z]+\.?$')
num_pattern = re.compile(r'^[\d]+$')

def wordFilter(word):
    if len(word) == 0:
        return False
    if cap_pattern.match(word):
        return False
    if num_pattern.match(word):
        return False
    return True

# Load the text file
words = sc.textFile('Shakespeare.txt').flatMap(lambda line: line.split(' ')).filter(lambda word: wordFilter(word))

# This is the tuple structure we're aiming for:
#    ((Word1, Word2), [(Word3a, Count3a), (Word3b, Count3b), ...])
withIndex = words.zipWithIndex()
the_words = withIndex.map(lambda x: (x[1], x[0]))
one_word_behind = the_words.map(lambda x: (x[0] + 1, x[1]))
two_words_behind = one_word_behind.map(lambda x: (x[0] + 1, x[1]))

# Join the lists with offset indexes to create the nice tuple list
print "Joining lists now"
word_groups = two_words_behind.join(one_word_behind).join(the_words)

def sumValues(list):
    valueDict = {}
    for l in list:
        if l in valueDict:
            valueDict[l] = valueDict[l] + 1
        else:
            valueDict[l] = 1
    return valueDict.items()

word_groups2 = word_groups.values().groupByKey().map(lambda x: (x[0], sumValues(list(x[1]))))

def getNiceTuple(cur_string, second, next_list):
    wwlist = []
    for ww in next_list:
        for i in range(0, ww[1]):
            wwlist.append(ww[0])
    choice = random.choice(wwlist)
    return (cur_string + " " + second, (second, choice))

# Generate 10 sentences
print "Generating 10 sentences now"
for i in range(0, 10):
    current_state = sc.parallelize(word_groups2.takeSample(False, 1, i)).map(lambda seq: getNiceTuple(str(seq[0][0]), seq[0][1], seq[1])).take(1)[0]
    for i in range (0, 18):
        current_state = getNiceTuple(current_state[0], current_state[1][1], word_groups2.lookup(current_state[1])[0])
    print current_state[0]