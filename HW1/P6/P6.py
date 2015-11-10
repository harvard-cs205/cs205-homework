# Markov Shakespeare
import numpy as np

# initiallizing spark
from pyspark import SparkContext, SparkConf
sc = SparkContext(conf=SparkConf())
sc.setLogLevel("ERROR")

'''
*******************
    FUNCTIONS
*******************
'''

'''
wordcounts
input: list of words
output: list of unique words and their counts

'''
def wordcounts(lst):
    # create dictionary of unique words and counts
    worddict = dict()
    for i in lst:
        if i in worddict:
            worddict[i] += 1
        else:
            worddict[i] = 1

    # make into list
    result = []
    for i in worddict.keys():
        result.append((i,worddict[i]))

    return result

'''
add
will use later in reduceByKey
'''
def add(x,y): return x + y

'''
makephrase
input: rdd of format ((w1,w2),[w3a,w3b,w3a,...])
output: sentence string
'''

def makephrase(rdd):
    count = 0 # counter for words in phrase
    it = 0 # iterations of restarting to kill it if there is something wrong

    while count < 20:
        # in case something goes wrong
        if it == 5:
            break

        # first 2 words
        if count < 2:
            # initialize phrase array
            phrase = []

            # select random key
            key = rdd.takeSample(True,1)[0][0]

            # put keys into phrase
            phrase.append(key[0])
            phrase.append(key[1])

            count += 2

        # next 18 words
        else:
            # choose next word
            # we can choose directly from the associations RDD b/c it
            # contains all instances of word3 so it is more likely to
            # choose words that appear more times. we don't need to
            # random sample from a distribution or anything because the
            # RDD I created is already biased by the counts
            nextword = np.random.choice(rdd.map(lambda x: x).lookup(key)[0])

            # if nextword is [] start again
            if nextword == []:
                count = 0
                it += 1
                continue

            # add new word it into phrase
            phrase.append(nextword)

            # next key
            key = (phrase[-2],phrase[-1])

            # increment
            count += 1

    # put it into a sentence string
    sentence = ''
    for i in phrase:
        sentence += i + ' '
    print sentence
    print


    return




'''
*********************************
    MAKING THE NECESSARY RDDS
*********************************
'''

# read in text file to a string called document
f = open('HW1/P6/Shakespeare.txt','r')
document = f.read()

# split into array of words
wordlist = document.split()

# parse through list to remove unwanted items
# wordlist is array of words in order
wordlist = [i for i in wordlist if not(i.isupper() or i.isdigit())]

# gives us all 3 word triples in order
# creates [(w1,w2,w3),(w1,w2,w3),...]
triples = []
for i in range(len(wordlist)-2):
    triples.append((wordlist[i],wordlist[i+1],wordlist[i+2]))

# make RDD of ((w1,w2),w3)
associations = sc.parallelize(triples)
associations = associations.map(lambda x: ((x[0],x[1]), [x[2]])).partitionBy(10)

# group RDD so we get ((w1,w2),[w3a,w3b,w3c,...])
associations = associations.reduceByKey(add)

# create RDD of format ((w1,w2),[(w3a,count),(w3b,count),(w3c,count),...])
associationswcounts = associations.mapValues(wordcounts)

# print 10 generated 20 word phrases
for i in range(10):
    makephrase(associations)
