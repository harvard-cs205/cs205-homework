import findspark
findspark.init('/home/lhoang/spark')

import pyspark
sc = pyspark.SparkContext(appName="spark1")


def get_gram(v):
    '''
    Converts an iterable into a sorted dictionary mapping from value to
    number of occurences.
    :param v: the iterable to convert
    '''
    D = {}
    for vv in v:
        if vv in D:
            D[vv] += 1
        else:
            D[vv] = 1
    a = []
    for k in D:
        a.append((k, D[k]))
    return sorted(a, key=lambda x: -x[1])


def is_number(s):
    '''
    Checks if the given string is a number.
    :param s: the string to check
    '''
    try:
        float(s)
        return True
    except ValueError:
        return False


def discard_word(s):
    '''
    Checks whether the given word should be discarded:
    - all uppercase
    - all numbers
    - all uppercase ending with a period.
    :param s: the string to check
    '''
    return s.isupper() or (s[:-1].isupper() and s.endswith(
        '.')) or is_number(s)


''' Main program '''
partition_size = 20
num_sentences = 10
num_words_per_sentence = 20

pg = sc.textFile('pg100.txt')

# Build intermediate index-to-word RDDs
i_w_1 = pg.flatMap(lambda l: l.split()).filter(
    lambda w: discard_word(w) == False).zipWithIndex().map(
    lambda wi: (wi[1], wi[0]))
i_w_2 = i_w_1.map(lambda kv: (kv[0] + 1, kv[1])).join(i_w_1)
i_w_3 = i_w_2.map(lambda kv: (kv[0] + 1, kv[1])).join(i_w_2)
model = i_w_3.map(lambda kv: (kv[1][0], kv[1][1][1]))

# Build model RDD mapping from word pairs to next likely words
# [ (k,v), ([(k, c), (k, c), . . .]) ]
model = model.groupByKey().mapValues(get_gram).partitionBy(
    partition_size).cache()

# Build sentences RDD
# [ (k,v), ([(k, c), (k, c), . . .], 'path') ]
sentences = sc.parallelize(model.takeSample(
    False, num_sentences, 10)).mapValues(lambda v: (v, '')).partitionBy(
    partition_size).cache()

# Main loop
for iteration in range(num_words_per_sentence):
    # get the next pair of words
    # [ (k, v), 'path' ]
    next_pairs = sentences.map(
        lambda kv: ((kv[0][1], kv[1][0][0][0]),
                    kv[1][1] + ' ' + kv[0][0]))

    # join pair of words with model so we can find next words
    # [ (k, v), ( [(k, c), (k, c), . . .], 'path' ) ]
    sentences = model.join(next_pairs).cache()

print sentences.mapValues(lambda kv: kv[1]).collect()
