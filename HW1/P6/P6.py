from pyspark import SparkContext
import numpy as np


def parsefilter(word):
    if len(word) == 0:
        return False
    if word.isdigit():
        return False
    if len(word) > 1 and word.isalpha() and word.isupper():
        return False
    if len(word) > 1:
        if word[-1]=='.' and word[0:-1].isalpha() and word[0:-1].isupper():
            return False
    return True

if __name__ == '__main__':
    sc = SparkContext("local", appName="Spark1")
    
    txtfile = sc.textFile('pg100.txt', use_unicode=False) #load text
    #txtfile = sc.textFile('test.txt', use_unicode=False) #load text
    words = txtfile.flatMap(lambda line: line.split(" ")).filter(parsefilter)
    
    word0 = words.zipWithIndex().map(lambda p: (p[1]+2, p[0]))
    word1 = word0.map(lambda p: (p[0]-1, p[1]))
    word2 = word1.map(lambda p: (p[0]-1, p[1]))
    w3gram = word0.join(word1).join(word2).values().map(lambda p: ( (p[0][0], p[0][1], p[1]), 1) ).reduceByKey(lambda a, b: a + b) # ( (w1, w2, w3), count  )
    w3gram_t = w3gram.map(lambda p:   ( (p[0][0], p[0][1]), [(p[0][2], p[1])] )   ) # ( (w1, w2), [(w3, count)]  )
    w2to3 = w3gram_t.reduceByKey(lambda v1, v2: v1+v2) #( (w1, w2), [ (w3a, c3a), (w3b, c3b),... ]  )
    
    samples = w2to3.takeSample(False, 10)
    result = []
    for start in samples:
        wordseq = [start[0][0], start[0][1]]
        while len(wordseq) < 20:
            curr2words = ( wordseq[-2], wordseq[-1] )
            nextpossible = w2to3.map(lambda x:x).lookup(curr2words)[0]
            nextprob = np.array([ t[1] for t in nextpossible], dtype='float')
            nextprob = nextprob / np.sum(nextprob)
            nextword = np.random.choice( [ t[0] for t in nextpossible], p=nextprob )
            wordseq.append(nextword)
        result.append( wordseq )
    
    for l in result:
        print l
    
    #print sorted(w2to3.collect(), key=lambda x: len(x[1]) )