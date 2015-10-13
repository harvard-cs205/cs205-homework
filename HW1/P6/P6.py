import findspark 
findspark.init()
from pyspark import SparkContext
import random

sc = SparkContext()
shakespeare = sc.textFile("shakespeare.txt")

#split words apart to be filtered
cleanup = shakespeare.map(lambda w: w.split())

#filters out unwanted words and returns a list of words to use
def banned(listness):
    passed = []
    for x in listness:
        if not x.isupper() and not x.isdigit() and not x.isspace() and not (x[:-1].isupper() and x.endswith('.')):
            passed.append(x)
    return passed

#the rdd of words to use
cleanedup = cleanup.flatMap(lambda w: banned(list(w)))
process = cleanedup.collect()

#make a list of each word with its following 2 words
bigList = []
for v in range(len(process[:-2])):
    smallList = []
    smallList.extend((process[v], process[v+1], process[v+2]))
    bigList.append(smallList)

#creates a RDD of three words lists and turns into a KV map with the first two
#words mapped to the last words and how many times they repeated
threeWords = sc.parallelize(bigList,100)
wordKV1 = threeWords.map(lambda x: ((x[0],x[1]), x[2]))
wordKV2 = wordKV1.groupBy(lambda (x,y): (x,y)).map(lambda (x,y): (x[0],(x[1],len(list(y)))))
finalwordKV = wordKV2.groupByKey().map(lambda (x,y): (x, list(y)))

#create a random phrases
def makePhrase():
    #Sudo-code since I ran out of time...

    #Take finalwordKV rdd and random number between 0 and length of rdd
    #    and select a key.
    #Add the first two words to a new list.
    #*Then in that key look at list of values and sum the total number of counts
    #    from all the word3's.
    #Then randomly select a number between 0 and the sum-1, and pick the index that 
    #    corresponds to it. 
    #        (ex. (w3a, 1), (w3b, 4), (w3c, 2))
    #        sum = 7
    #        random number = 4
    #        index will be w3b
    #next find the key that has the second word of the key and the selected value word.
    #repeat from * 19 times
    #return list of words
    
for x in range(10):
    makePhrase()
    
    


