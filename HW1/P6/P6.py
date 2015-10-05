# list of the words we will work on keeping the order
data = []
# read data
f = open("Shakespeare.txt")
for line in f:
    for w in line.split():

        if not w[:-1].isdigit():
            if not w.isupper():
                if not (w[:-1].isupper() and w[-1]=='.'):
                    data.append(w)
# create a single line from all the data
longline = ' '.join(data)
# generator for the tuple of words
def chunks(l):
    for i in xrange(len(l)-2):
        yield l[i:i+3]
# create a RDD
superline = sc.parallelize([longline])
# get groups of 3 words
w_tuple = superline.flatMap(lambda line: [((chunk[0],chunk[1],chunk[2]), 1) for chunk in chunks(line.split())])
# create the distribution for the Markov 2nd order
tuple_dist = w_tuple.reduceByKey(lambda x, y: x + y)
# the key is the pair of words
# the value is a pair with the next word and it's weight to be sampled
RDD_toReduce = tuple_dist.map(lambda (k,v): ((k[0],k[1]),(k[2],v)))
# get a list of possibly sampled pair as value
RDD = RDD_toReduce.groupByKey().mapValues(list)
# sampling function for a word
def sample_next_word(RDD,sentence):
    import numpy as np
    # find all the words and weights associated with this pair of word
    list1 = RDD.map(lambda x:x).lookup((sentence[-2],sentence[-1]))
    # get all the weights and create a distribution
    weight = np.array([float(v[1]) for v in list1[0]])
    # normalize
    weight /= float(np.sum(weight))
    # create the list of possible words
    sample = [v[0] for v in list1[0]]
    # sample it
    sentence.append(np.random.choice(sample,p = weight))
    return sentence

# test initialization
sentence = ['Now','is']
for k in range(18):
    sentence=sample_next_word(RDD,sentence)

# samples k sentences
def create_sentences(k):
    sentences = []
    for i in range(k):
        # As said in class for the 2 first words we just use takeSample
        words12 = RDD.map(lambda (k,v):k).takeSample(False,1)[0]
        sentence = [words12[0], words12[1]]
        for k in range(18):
            sentence=sample_next_word(RDD,sentence)
        sentences.append(sentence)
    return sentences
