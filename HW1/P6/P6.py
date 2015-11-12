# File used with Spark on ipython

import numpy as np
import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="Spark1")

# ### Formating the data
# rdd format: (word)
lines = sc.textFile('Shakespeare.txt')
words = lines.flatMap(lambda x: x.split(" "))

# Filtering the words based on the rule
words = words.filter(lambda x: sum([not v.isupper() for v in x[:-1]]) and
                     sum([not v.isdigit() for v in x]))

# Set the rdd in the form (index, correct_word)
words = words.zipWithIndex().map(lambda x: (x[1], x[0]))

# Strategy to group by triplet:
# Increase the index of each word by 1 and also by 2 in two rdds
# then joining them together on the key groups the group of 3 consecutives
# words
first_words = words.map(lambda x: (x[0] + 2, x[1]))
second_words = words.map(lambda x: (x[0] + 1, x[1]))

# Join the occurent double of words together
# rdd format: (index, [word1, word2])
double_words = first_words.join(second_words).map(lambda x: (x[0], list(x[1])))

# Join the double with each of his successor to build the occurent triplets
# rdd format: ((word1, word2), word3)
triplet_words = words.join(double_words).map(lambda x: (list(x[1])[1],
                                             list(x[1])[0]))


def count_list(x):
    '''
    Function used to build the asked rdd format in the join.
    '''
    key = x[0]
    value = list(x[1])
    counter = [(w, value.count(w)) for w in list(set(value))]
    return key, counter

# Building the requested rdd
# rdd format: ((word1, word2), [(word3a, count3a), (word3b, count3b), ...])
occurence_words = triplet_words.groupByKey().map(lambda x: count_list(x))

# ### Text generation


def next_word(counter):
    '''
    Return the next word based on the next word transition given in counter.
    The probability of transition is computed on the square of the transition
    to introduce a bias on the count.
    '''
    t = [c[1] for c in counter]
    t2 = np.multiply(t, t)
    transition_proba = (1. / sum(t2)) * t2
    next_stage = np.random.multinomial(1, transition_proba)

    return counter[next_stage.argmax()][0]


def Shakespeare_quill(occurence_graph, s=10, w=20):
    '''
    Function to make Shakespeare's quill alive, generate s sentences of w words
    Args:
    s: number of sentences
    w: number of words per sentence
    Return:
    words_list: the list of the words
    '''
    # current: current double word
    # counter: occurences of the next word after current
    current, counter = occurence_graph.takeSample(True, 1)[0]
    words_list = list(current)

    for i in xrange(0, s):
        for j in xrange(0, w):
            # Compute the next word
            word = next_word(counter)
            # Debug mode
            # print('newt word is {}'.format(word))
            # Add the next word to the current line
            words_list.append(word)
            # Set the next tuple
            current = (current[1], word)
            # Debug mode
            # print('current word is {}'.format(current))
            # Retrieve the counter information of the next tuple
            counter = occurence_graph.lookup(current)[0]
            # Debug mode
            # print('counter is {}'.format(counter))
    return words_list

# ### To generate a text and print it


def print_text(words_list, s=10, w=20):
    for i in xrange(0, s):
        print(' '.join(words_list[i * w:(i+1)*w]))

words_list = Shakespeare_quill(occurence_words)
print_text(words_list, s=10)
