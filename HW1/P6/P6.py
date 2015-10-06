import findspark
findspark.init()

from operator import add
import re

import numpy as np
import pyspark

sc = pyspark.SparkContext()

sc.setLogLevel('WARN')

# regular expression that matches words that:
#   * contain only numbers
#   * contain only letters which are capitalized
#   * contain only letters which are capitalized and end with a period
regex = re.compile(r"(?:\A[0-9]+\Z)|(?:\A[A-Z]+\Z)|(?:\A[A-Z]+\.\Z)")
words = sc.textFile("shakespeare.txt").flatMap(lambda x: x.split()).filter(lambda x: not regex.match(x)).collect()

triples = []

# add every sequential 3-word triple to list
for i in xrange(len(words) - 2):
    triples.append((words[i], words[i + 1], words[i + 2]))

counts = sc.parallelize(triples)


def calculate_probabilities(tup):
    (word1, word2), next_words = tup
    words, counts = zip(*next_words)
    probs = np.array(counts) / float(sum(counts))
    next_words = zip(words, probs)
    return (word1, word2), next_words

# take triples and turn into an RDD with
# ((word1, word2), [(word3_1, prob_word3_1), (word3_2, prob_word3_2), ...]
counts = counts.map(lambda x: (x, 1)).reduceByKey(add).map(lambda x: ((x[0][0], x[0][1]), [(x[0][2], x[1])])).reduceByKey(add).map(calculate_probabilities)


def get_next_word(word1, word2):
    """
    Given word1 and word2, finds a third word according
    to the probabilities calculated above.
    """
    next_words = counts.lookup((word1, word2))[0]
    words, probs = zip(*next_words)
    next_word = np.random.choice(words, p=probs)
    return next_word


def make_sentence(sentence_length):
    """
    Makes a sentence starting from a random
    word1, word2 pair, of length sentence_length.
    """
    seed = counts.takeSample(False, 1)[0]
    sentence = list(seed[0])

    for i in xrange(sentence_length - 2):
        word1, word2 = sentence[-2], sentence[-1]
        next_word = get_next_word(word1, word2)
        sentence.append(next_word)

    return " ".join(sentence)

# make 10 sentences of length 20 words
for _ in xrange(10):
    print make_sentence(20)
