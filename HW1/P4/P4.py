from pyspark import SparkContext
sc=SparkContext()

data=sc.textFile('/home/memory/cs205-homework/HW1/P4/source.csv')
rdd=data.map(lambda x: x.split("\",\""))
rdd1=rdd.map(lambda x: (x[1].replace("\"",""),x[0].replace("\"",""))).groupByKey().mapValues(list)

def f(x):
    z=[]
    for j in range(len(x)):
        z.append((x[j], [x[i] for i in range(len(x)) if i!=j]))
    return z

final=rdd1.values().flatMap(f).reduceByKey(lambda x,y: list(set(x+y))).cache()
