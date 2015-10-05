import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import string

import pyspark
sc = pyspark.SparkContext()
sc.setLogLevel("ERROR")

#Checks if words is all capitalised and has a punctuation of any kind at its end:
def capandpunc(w):
    punc = [",",".",";",":","!","?"]
    res1 = all(l.isupper() for l in w[-(len(w)-1)])
    res2 = (w[(len(w)-1)] in punc)
    return (res1 and res2)

#Reading the source file:
text = sc.textFile("pg100.txt") 

#Splitting into words:
words = text.flatMap(lambda line: line.split())

#Cleaning:
words = words.filter(lambda w: not all(l.isdigit() for l in w))
words = words.filter(lambda w: not all(l.isupper() for l in w))
words = words.filter(lambda w: not capandpunc(w))

#Indexing the words and putting index as keys:
words = words.zipWithIndex().map(lambda (K,V): (V,K))

#Creating 3 RDDs with different indices to create triples:
words1 = words.filter(lambda (K,V): K!=(826800-1))
words2 = words.map(lambda (K,V): (K-1,V)).filter(lambda (K,V): K>=0)
words3 =  words.map(lambda (K,V): (K-2,V)).filter(lambda (K,V): K>=0)

#Creating the triples:
tutriple = words1.union(words2).union(words3).reduceByKey( lambda u,v: [u]+[v]).filter(lambda x: x[0]<(826800-2))
tutriple = tutriple.map(lambda x: (((x[1][0][0],x[1][0][1]),x[1][1]),1))

# Are in the form (((W_1,W_2),W_3),1) where 1 is the count

#Count the number of triples:
tutriple = tutriple.reduceByKey(lambda u,v: u+v)

#Changing the format of the RDD's elements to the desired one in the pset:
tutriple = tutriple.map(lambda (K,V): (K[0],(K[1],V)))
result = tutriple.groupByKey().map(lambda (K,V):(K,list(V)))

#sanity check:
result.lookup(('Now','is'))[0]

#Creating an RDD with only the keys in order to sample from it:
tuples = result.keys()

#Generating next word function, that takes for inuput the first 2 words in a tuple form and the model RDD:
def generateword(pair,rdd):
    wordsandcount = rdd.lookup(pair)[0]
    words = [i[0] for i in wordsandcount]
    prob = [float(i[1]) for i in wordsandcount]
    prob = [i/sum(prob) for i in prob]
    
    return np.random.choice(words,p=prob)
    
# Generating the 20 sentences:
for j in range(20):
    words_to_print = list(tuples.takeSample(False,1)[0])
    for i in range(18):
        words_to_print.append(generateword((words_to_print[i],words_to_print[i+1]),result))
    print ' '.join(words_to_print)+"\n"