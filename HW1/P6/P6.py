import findspark
findspark.init()

import re

import numpy as np
import pyspark

sc = pyspark.SparkContext()

sc.setLogLevel('WARN')

regex = re.compile(r"(?:\A[0-9]+\Z)|(?:\A[A-Z]+\Z)|(?:\A[A-Z]+\.\Z)")
words = sc.textFile("shakespeare.txt").flatMap(lambda x: x.split()).filter(lambda x: not regex.match(x)).collect()

triples = []

for i in xrange(len(words) - 2):
    triples.append((words[i], words[i + 1], words[i + 2]))

counts = sc.parallelize(triples)


def calculate_probabilities(tup):
    (word1, word2), next_words = tup
    words, counts = zip(*next_words)
    probs = np.array(counts) / float(sum(counts))
    next_words = zip(words, probs)
    return (word1, word2), next_words


counts = counts.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y).map(lambda x: ((x[0][0], x[0][1]), [(x[0][2], x[1])])).reduceByKey(lambda x, y: x + y).map(calculate_probabilities)


def get_next_word(word1, word2):
    next_words = counts.lookup((word1, word2))[0]
    words, probs = zip(*next_words)
    next_word = np.random.choice(words, p=probs)
    return next_word


def make_sentence(sentence_length):
    seed = counts.takeSample(False, 1)[0]
    sentence = list(seed[0])

    for i in xrange(sentence_length - 2):
        word1, word2 = sentence[-2], sentence[-1]
        next_word = get_next_word(word1, word2)
        sentence.append(next_word)

    return " ".join(sentence)
