
import pyspark
from pyspark import SparkContext
sc = SparkContext()
from collections import Counter
import random


text = sc.textFile('shakespeare.txt')
# parse text and filter certain words
w1 = text.flatMap(lambda l: l.split(" ")).filter(lambda x: x != '')
w2 = w1.filter(lambda x: x.isdigit() == False)
w3 = w2.filter(lambda x: x.isupper() == False)
word = w3.filter(lambda x: x.endswith('.') == False and x[:-1].isupper() == False)

# build an rdd of the form ((word1, word2), word3)
r1 = word.zipWithIndex()
r2a = r1.map(lambda x: (x[1], x[0]))
r2b = r1.map(lambda x: (x[1]+1, x[0]))
r2c = r1.map(lambda x: (x[1]+2, x[0]))
j1 = r2a.leftOuterJoin(r2b).map(lambda x:( x[0], (x[1][1], x[1][0])))
j2 = j1.leftOuterJoin(r2c).map(lambda x:(x[0], (x[1][1], x[1][0])))
r2 = j2.filter(lambda x: x[0] >= 2 and x[0] <= 754954)
r3 = r2.map(lambda x: x[1])
r4 = r3.map(lambda x: ((x[0], x[1][0]), x[1][1])) #((word1,word2), word3)
r5 = r4.groupByKey().mapValues(list)

# count the number of word3s
def make_count(x):
    C = Counter(x)
    l = [[k,]*v for k,v in C.items()] # [[w1,w1],[w2,w2,w2]]
    wcount = [(x[0], len(x)) for x in l]
    return wcount

# create red of the form ((word1,word2), [(word3a,count3a),(word3b,count3b)]) 
r_final = r5.map(lambda x: (x[0], make_count(x[1])))

# produce sentence for one random word
def produce_sentence(rdd):
    word = rdd.sample(False, 0.00004, None).collect()[0]
    wordlist = []
    prev = word[0][1]
    wordlist.append(word[0][0])
    wordlist.append(prev)
    w = word[1]
    for j in range(18):  # generate 18 words
        l = []
        sum = 0
        for i in range(len(w)): #construct PMF
            sum = sum + w[i][1]
            l.append(sum)
        rand = random.random()
        # find the most likely word3                                                                                           
        for i in range(len(w)):
            if i == 0 and rand < l[i]/sum*1.0:
                third = w[i][0]
                wordlist.append(third)
                w = rdd.lookup((prev, third))[0]
                prev = third
            elif rand >= l[i-1]/sum*1.0 and rand < l[i]/sum*1.0:
                third = w[i][0]
                wordlist.append(third)
                w = rdd.lookup((prev, third))[0]
                prev = third
    return wordlist

# print the 10 sentences
for i in range(10):
    sentence = produce_sentence(r_final)
    for s in sentence:
        print s
        print (" ")
    print "\n"


    


