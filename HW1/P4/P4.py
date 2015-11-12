import findspark 
findspark.init()
from pyspark import SparkContext
import bfs from P4_bfs

sc = SparkContext()

comicness = sc.textFile("comics.csv")

char_iss_messy = comicness.map(lambda x: x.split('"'))
char_rdd = char_iss_messy.map(lambda (l,m,n,o,p): (o,m))
iss_rdd = char_iss_messy.map(lambda (l,m,n,o,p): (m,o))

characterMap = char_rdd.groupByKey().map(lambda x: (x[0], list(x[1])))
issueMap = iss_rdd.groupByKey().map(lambda x: (x[0], list(x[1])))

def mapout(charlist):
    newlist = []
    charV = []
    for i in range(len(charlist)):
        charK = charlist[i]
        for a in charlist:
            if a != charlist[i]:
                charV.append(a)
        newlist.append([charK,charV])
        charV = []
        
    return newlist

#maps character to other characters in the same comic issue
graph = characterMap.flatMap(lambda x: (mapout(x[1]))) 

captainA = bfs("CAPTAIN AMERICA", graph)
missT = bfs("MISS THING/MARY", graph)
orwL = bfs("ORWELL", graph)


print len(captainA)
print len(missT)
print len(orwL)






