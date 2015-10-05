
import pyspark
from pyspark import SparkContext
sc = SparkContext()


# read in data
data = sc.textFile('source.csv')

# build graph representation
# reform data and group by comic issue
rdd1 = data.map(lambda r:(r.split('\",\"')[1].replace('"',''), r.split('\",\"')[0].replace('"','')))
rdd2 = rdd1.groupByKey().mapValues(list)

# split values above and create a graph with each character as key
def pair_char(r):
    l = []    
    for i in range(len(r)):
        x = r[i]
        y = []
        for j in range(len(r)):
            if j != i:
                y.append(r[j])    
        l.append((x,y))
    return l


rdd3 = rdd2.values().flatMap(pair_char)

# group by key for all comic books
rdd_g = rdd3.reduceByKey(lambda x,y: list(set(x+y)))











