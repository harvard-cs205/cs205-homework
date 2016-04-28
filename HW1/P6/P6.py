import re
from pyspark import SparkContext
sc=SparkContext()

lines=sc.textFile('/home/memory/cs205-homework/HW1/P6/pg100.txt')
words=lines.flatMap(lambda x: x.split())

words_filter=words.filter(lambda x: re.match(r"^(\d+|[A-Z]+\.*)$",x) is None)

RDD=words_filter.zipWithIndex()
RDD1=RDD.map(lambda x: (x[1],x[0]))
RDD2=RDD1.map(lambda x: (x[0]-1,x[1]))
RDD3=RDD1.map(lambda x: (x[0]-2,x[1]))
RDD1_RDD2=RDD1.join(RDD2)
together=RDD1_RDD2.join(RDD3)

key=together.values().map(lambda x: ((x[0][0],x[0][1],x[1]),1))

final=key.reduceByKey(lambda x,y: x+y).map(lambda x: ((x[0][0],x[0][1]),(x[0][2],x[1]))).groupByKey().mapValues(list).cache()

x=final.keys().takeSample(False,10,)

for i in range(10):
    z=[x[i][0],x[i][1]]
    for j in range(18):
        find=(z[j],z[j+1])
        y=final.map(lambda r: r).lookup(find)
        y[0].sort(key=lambda r: r[1],reverse=True)
        z.append(y[0][0][0])
    print " ".join(z)
