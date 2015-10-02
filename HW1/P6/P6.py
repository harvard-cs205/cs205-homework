from collections import Counter
from pyspark import SparkContext, SparkConf
import random

#Setup
conf = SparkConf().setAppName("shakespeare").setMaster("local[*]")
sc = SparkContext(conf=conf)

#Split and filter
lines = sc.textFile('pg100.txt')
all_words = lines.flatMap(lambda line: line.split())
words = all_words.filter(lambda x: not (x.isdigit() or (x[:-1].isalpha() and x[:-1].isupper() and (x[-1:] == '.' or (x[-1:].isalpha() and x[-1:].isupper())))))

def offset(KV):
    return (KV[0] - 1, KV[1])

#Markov model
words_indices = words.zipWithIndex().map(lambda KV: (KV[1], KV[0]))
words_indices_offset1 = words_indices.map(offset)
words_indices_offset2 = words_indices_offset1.map(offset)
pairs = words_indices.join(words_indices_offset1)
triplets = pairs.join(words_indices_offset2).values().mapValues(lambda v: [v])
markov_chain = triplets.reduceByKey(lambda x,y: x + y).mapValues(lambda vs: Counter(vs)).mapValues(lambda vs: [(key, vs[key]) for key in vs]).cache()

#Taken from http://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
def weighted_choice(choices):
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
          if upto + w >= r:
             return c
          upto += w
    assert False, "Shouldn't get here"
    
#Generate Sentences
sentences = []
starts = markov_chain.takeSample(False, 10)
for start in starts:
    current_tuple = start
    shakespeare_sentence = current_tuple[0][0] + ' ' + current_tuple[0][1]
    for j in range(2, 20):
        next_word = weighted_choice(current_tuple[1])
        current_tuple_precursor = markov_chain.filter(lambda KV: KV[0] == (current_tuple[0][1], next_word))
        current_tuple = current_tuple_precursor.first()
        shakespeare_sentence = shakespeare_sentence + ' ' + next_word
    sentences.append(shakespeare_sentence)
print sentences
