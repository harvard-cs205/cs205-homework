from __future__ import division
import pyspark
import random
from operator import add

def words_we_want(w):
    dialog_marker = w.endswith('.') and w[:-1].isupper()
    return (not w.isdigit()) and (not w.isupper()) and (not dialog_marker)

# from http://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
def weighted_choice(choices):
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w >= r:
            return c
        upto += w

def choose_next_word(last_w, so_far, next_ws):
    next_w = weighted_choice(next_ws)
    return ((last_w, next_w), so_far + [next_w])

if __name__ == "__main__":
    # initialize spark context
    conf = pyspark.SparkConf().setAppName("P6").setMaster("local[*]")
    sc = pyspark.SparkContext(conf=conf)

    # get words from text file
    shakespeare = sc.textFile('Shakespeare.txt')
    words = shakespeare.flatMap(lambda l: l.split(' ')).filter(words_we_want).zipWithIndex().cache()

    # build Markov table
    # first, get all sets of 3 consecutive words
    words0 = words.map(lambda (k,v): (v,k)).partitionBy(8).cache()
    words1 = words.mapValues(lambda v: v-1).map(lambda (k,v): (v,k)).partitionBy(8).cache()
    words2 = words.mapValues(lambda v: v-2).map(lambda (k,v): (v,k)).partitionBy(8).cache()
    consec_words = words0.join(words1).join(words2).values().map(lambda x: (x,1))

    # then count how many of each
    consec_counts = consec_words.reduceByKey(add)

    # and group by first two words of triplet
    markov_table = consec_counts.map(lambda (((w1,w2),w3),c): ((w1,w2),(w3,c))).groupByKey().cache()

    # check that we've done this correctly
    #print list(markov_table.map(lambda x: x).lookup(('Now','is'))[0])

    # sample from Markov table
    frac = 50/markov_table.count()
    sample = markov_table.sample(False, frac).keys()
    sentences = sample.zip(sample.map(lambda ws: list(ws)))

    # generate sentences of length 20
    length = 20
    for i in range(length - 2):
        sentences = sentences.join(markov_table).map(lambda ((w1,w2),(lst,w3s)): choose_next_word(w2,lst,w3s))

    for sentence in sentences.collect():
        print ' '.join(sentence[1])
