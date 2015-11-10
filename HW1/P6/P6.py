import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext()

from collections import Counter # for 3rd word counts
import numpy as np 				# for sampling
import time 					# for benchmarking

def quiet_logs(sc):
	''' Shuts down log printouts during execution (thanks Ray!) '''
	logger = sc._jvm.org.apache.log4j
	logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)
	logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)
	logger.LogManager.getLogger("amazonaws").setLevel(logger.Level.WARN)

quiet_logs(sc)


''''''''''''''''''''''''''
''' format word corpus '''
''''''''''''''''''''''''''
n_part      = 10 # num partitions
shakespeare = sc.textFile("Shakespeare.txt", n_part)

def excluded(w):
	''' checks if word meets exclusion criteria ''' # exclude if:
	all_upper = w.isupper() 						# all uppercase
	all_upper_plus_period = False					
	if w[-1]==".": 
		all_upper_plus_period = w[:-1].isupper()	# all uppercase ending in period
	all_numbers = w.isdigit()						# all digits
	return all_upper or all_upper_plus_period or all_numbers

def format_corpus():
	''' splits corpus into words, filters by exclusion criteria'''
	return lambda corpus: [word for word in corpus.split(" ") if word and not excluded(word)]

def countit(wlist):
	''' returns (word,frequency) tuple '''
	return Counter(wlist).most_common()

''' we want our corpus in the format: ( (word1,word2), [(word3a,word3a_freq),(word3b,word3b_freq),...]) 
	- we can do this by first getting ( (word1,word2), word3 ) tuples.
	- make 3 corpora: first corpus unchanged, second corpus offset start index by 1,  third corpus offset start by 2
	- then join all 3 corpora into one - this gives us ( (word1,word2), word3 ) tuples. 
	- from there we can groupByKey and format.
'''

words  = shakespeare.flatMap(format_corpus()).zipWithIndex().partitionBy(n_part).cache()
wct    = words.count()
wlist1 = words.filter(lambda x: x[1] < wct-2).map(lambda x: tuple(reversed(x))).partitionBy(n_part)
wlist2 = words.filter(lambda x: (x[1] > 0) and (x[1] < wct-1)).map(lambda x: tuple(reversed(x))).map(lambda x: (x[0]-1,x[1])).partitionBy(n_part)
wlist3 = words.filter(lambda x: x[1] > 1).map( lambda x: tuple(reversed(x)) ).map(lambda x: (x[0]-2,x[1])).partitionBy(n_part)
rdd    = (wlist1.join(wlist2, numPartitions=n_part)
				.join(wlist3, numPartitions=n_part)
				.map(lambda x: x[1])
				.partitionBy(n_part)
				.groupByKey(numPartitions=n_part)
				.map(lambda x: ( x[0],countit(x[1]) ), preservesPartitioning=True)
				.cache()
	   )
	
''''''''''''''''''''
''' markov model '''
''''''''''''''''''''

def get_third(third_words):
	''' creates empirical PMF from 3rd word counts, draws one word by sampling '''
	words,freqs = zip(*third_words)
	pmf         = np.array(freqs)/float(sum(freqs)) # PMF for observations
	third       = np.random.choice(words,p=pmf)     # draw word from PMF
	return third

n_samp = 10
n_iter = 20
starts = rdd.takeSample(False,n_samp)       # draw n_samp (first_word,second_word) tuples

def start_filter(s):
	return lambda x: x[0] in [w[0] for w in s]


timeA=time.time()

start  = rdd.filter( start_filter(starts) ) # filter rdd down to 10 starts
phrase = (start.zipWithIndex()              # initialize phrase as (idx, (word1,word2))
			   .map( lambda x: tuple(reversed(x)) )
			   .map( lambda x: (x[0],[el for el in x[1][0]]) )
			   .partitionBy(n_part)
		  )

for i in range(n_iter):   # we want n_iter-word phrases (target:20)
	if i:                 # after initial round, build start based on prior keys
		start = (rdd.join(key.map(lambda x: (x,['',0])).partitionBy(n_part), numPartitions=n_part) # add dummy [(word3,freq)] to key for join
					 .map(lambda x: (x[0],x[1][0]), preservesPartitioning=True)      # drop dummy (necessary?)
					 .partitionBy(n_part)
					 .cache()
				)
		
	key    = (start.map(lambda x: (x[0][1], get_third(x[1]))) # draw word3 from PMF, build new key
				   .partitionBy(n_part)
				   .cache()
			  )
	phrase = (phrase.union( key.zipWithIndex()                      # union new word3 to phrase 
							   .map( lambda x: tuple(reversed(x)) ) # make index into key for union
							   .partitionBy(n_part)
							   .map( lambda x: (x[0],x[1][1]), preservesPartitioning=True ) # take only word3
						  )
					.reduceByKey(lambda a,b: a+[b], numPartitions=n_part) # reduce to list of phrase words
			)
	
print phrase.map(lambda x: ' '.join(x[1])).collect() # print each phrase as string (space separator)

timeB=time.time()
print "total time:",timeB-timeA

# ~150s

