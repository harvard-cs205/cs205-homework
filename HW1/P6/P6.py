import re
import random

def clean(s):
    regPattern = ['(^[0-9]+$)', '^[A-Z]+$', '^[A-Z]+\.$']
    regObj = [re.compile(r) for r in regPattern]

    ret = []
    s = s.strip().split()
    for w in s:
        w = w.strip()
        rTests = [(r.match(w) == None) for r in regObj]
        if False not in rTests:
            ret.append(w)
    return ret

# clean("123 123. foo bar. BAR. 1 ABC")

def sepTriple(((k),v)):
    print k, v
    return ((k[0], k[1]), [(k[2],v)])

print 'Cleaning input...'

allWords = []
with open('pg100.txt','r') as f:
    for line in f:
        allWords += clean(line)

print 'Creating RDD...'

rddList = []
numP = 2
forW = 3

for i in xrange(0, len(allWords)-forW):
    rddList.append( tuple(allWords[i:i+forW]) )

RDD = sc.parallelize(rddList, numP)

RDD = RDD.map( lambda (v): (v, 1) )
RDD = RDD.reduceByKey(lambda a, b: a+b)
RDD = RDD.map( sepTriple )\
            .reduceByKey( lambda a,b: a+b)\
            .cache()

def randPick(words):
    norm = 0.
    r = random.uniform(0.,1.)
    for k,v in words:
        norm += v
    for k,v in words:
        r -= float(v) / norm
        if r < 0:
            return k
    return "ERROR"

# indexed = RDD.zipWithIndex()
# print indexed.take(10)

print 'Generating random words...'

for i in xrange(10):
    phrase = []
    first = RDD.takeSample(True, 1, None)[0]
    phrase += list(first[0])
    print 'Starting with', phrase
    for j in xrange(18):
        look = RDD.filter(lambda ( (k), v ): (k == tuple(phrase[-2:]))).take(1)

        if len(look) != 1:
            print "ERROR: no results returned, selecting random word"
            look = RDD.take(1) # Take a random word
        ((k),v) = look[0]
        phrase.append(randPick(v))

    print ' '.join(phrase)
