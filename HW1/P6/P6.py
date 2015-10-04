# (c) L.Spiegelberg 2015
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf

import urllib
import numpy as np
from scipy import stats
import re
import os
import sys

import matplotlib.ticker as ticker   
import seaborn as sns
sns.set_style("whitegrid")

def downloadFiles():
	# load textlist from net
	filename = "pg100.txt"

	url = "http://www.gutenberg.org/cache/epub/100/pg100.txt"

	wordfile = urllib.URLopener()
	wordfile.retrieve(url, filename)

	if not os.path.isfile('Shakespeare.txt'):
		# the texts from shakespeare are contained in lines 182 - 124367
		# create the file Shakespeare.txt using the line range from above
		# if all is ok, this should return True, True
		with open(filename, 'r') as f:
		    lines = f.readlines()
		    lines = lines[181:124367]
		    with open('Shakespeare.txt', 'w') as f2:
		        f2.writelines(lines)
		    print f2.closed


# Pardon the naming, but I couldn't resist!
def loadShakespearsBrain(context, filename):
    words = context.textFile(filename)
    
    # split lines into words (after whitespace)
    # before splitting replace first some escape characters with white space
    rdd = words.flatMap(lambda line: line.replace('\n', ' ').replace('\r', ' ').split(' ')).cache()
    
    # now perform filtering on the words
    # to develop the regular expression this awesome tool was used https://regex101.com

    # (1) filter out words that only contain numbers
    # (2) filter out words for which all letters are capitalized
    # (3) filter out words that contain letters only and end with a period

    pattern1 = re.compile(ur'([0-9]+\.?)|\.')  # use only a simple RE for numbers here (numbers in the shakespeare text are formateed as 1., 2., ...)
    pattern2 = re.compile(ur'(\b[A-Z]+\.?\b)|\.') # ==> do (2) & (3) together in one regex!

    # filter does not change the order
    rdd = rdd.filter(lambda x: pattern1.match(x) is None).filter(lambda x: pattern2.match(x) is None).filter(lambda x: len(x) > 0)
    
    # use index to identify the neighbored words!
    rdd = rdd.zipWithIndex()
    rdd = rdd.flatMap(lambda x: [(x[1], (0, x[0])), (x[1] - 1, (1, x[0])), (x[1] - 2, (2, x[0]))])
    rdd = rdd.groupByKey()

    # map (keyA, (keyA, wordA), (keyA + 1, wordA+1), (keyA + 2, wordA+2)) to (wordA, wordA+1, wordA+2)
    fun = lambda x: list(x[1])
    rdd = rdd.map(fun)

    # previous mapping created some tuples, that do not have length 3 at the beginning / end of the text
    # ==> get rid of them!
    rdd = rdd.filter(lambda x: len(x) == 3)

    # count number of combinations (key is the 3-word combination)
    rdd = rdd.map(lambda x: ((x[0][1], x[1][1], x[2][1]), 1))
    rdd = rdd.reduceByKey(lambda x, y: x + y)

    # remap form
    rdd = rdd.map(lambda x: ((x[0][0], x[0][1]), (x[0][2], x[1])))

    # final group by key
    rdd = rdd.groupByKey().map(lambda x: (x[0], sorted(list(x[1])) )) \
             .sortBy(lambda x: x[0])
    
    return rdd

# phrase generator
def generatePhrase(num_of_words, rdd):
	phrase = ''
	for i in range(0, num_of_words - 1):

	    if i == 0:
	        cur_sample = rdd.takeSample(False, 1)[0]
	        phrase = cur_sample[0][0] + ' ' + cur_sample[0][1]
	    else:
	        cur_sample = (cur_phrase, rdd.map(lambda x:x).lookup(cur_phrase)[0])   

	    cur_phrase = cur_sample[0]
	    nextwordlist = cur_sample[1]

	    # to choose the next word an next tuple, we have to draw from the random distribution that is described by the word lost
	    # this can be done manually (like here) or using scipy.stats i.e.

	    xk = np.arange(0, len(nextwordlist))
	    pk = np.array([w[1] for w in nextwordlist])
	    # normalize probabilities
	    pk = np.divide(pk, np.sum(pk) * 1.0)
	    word_distribution = stats.rv_discrete(name='word_distribution', values=(xk, pk))

	    # draw sample
	    index = word_distribution.rvs(size=1)[0]

	    # the next word is now at position index of the list
	    next_word = nextwordlist[index][0]

	    phrase += ' ' + next_word
	    cur_phrase = (cur_phrase[1], next_word)
	    
	return phrase


# program logic
def main(argv):

	# setup spark
	conf = SparkConf().setAppName('MarkovShakespeare')
	sc = SparkContext(conf=conf)

	# lazy load files
	downloadFiles()

	# transform txt file via Spark
	rdd = loadShakespearsBrain(sc, 'Shakespeare.txt')

	# generate 10 phrases
	num_phrases = 10
	num_of_words = 20
	with open('P6.txt', 'wb') as f:
		for i in range(0, num_phrases):
			phrase = generatePhrase(num_of_words, rdd)

			print('%02d: %s' % (i+1, phrase))
			f.write('%02d: %s\n' % (i+1, phrase))

# avoid to run code if file is imported
if __name__ == '__main__':
	main(sys.argv)
