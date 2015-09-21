from pyspark import SparkContext
import pdb

sc = SparkContext()
textFile = sc.textFile("EOWL_words.txt")
sortedTf = textFile.map(lambda word: (''.join(sorted(word)),word))
#pdb.set_trace()
groupedTf = sortedTf.groupByKey().map(lambda x: (x[0],len(list(x[1])),list(x[1])))
# maxTf = groupedTf.reduce(lambda a,b: a[0] if (a[0]>b[0]) else b[0])
# maxTf = groupedTf.reduce(lambda a,b: a+b)
#maxTf.take(10)
r=groupedTf.map(lambda x:(x[1],x[0],x[2]))
v=r.reduce(lambda a,b:max(a,b))
print v