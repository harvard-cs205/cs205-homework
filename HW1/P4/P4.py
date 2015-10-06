__author__ = 'xiaowen'
import findspark
findspark.init('/home/xiaowen/spark')
import pyspark
import itertools
from P4_bfs import bfs

# build sparkcontext object
sc = pyspark.SparkContext(appName="P4")

#  read data, convert it to RDD and split each lines by ','
items = sc.textFile('/home/xiaowen/cs205-homework/HW1/P4/source.csv')
items = items.map(lambda x: x.split('","'))

# Create pairs RDD (K,V) where volume is key and character is value
items_rdd = items.map(lambda x: (x[1].replace('"', '').strip(), x[0].replace('"', '').strip()))

# group the items_rdd by key(volume)
grouped_rdd = items_rdd.groupByKey().map(lambda x: (x[0], list(x[1])))


# convert the grouped_rdd to the graph we need

# function with a list as input return a dict with each element as key and other elements as value
def list_to_dict(x):
    result = []
    for i in range(len(x)):
        tmp = x[:i] + x[i + 1:]
        result.append((x[i], tmp))
    return result


# create a RDD with each char as key and the char in the same volume as value
graph = grouped_rdd.flatMap(lambda x: list_to_dict(x[1]))

# group graph_1 by key so that we could get all the edges of each node
graph = graph.groupByKey()

# merge all the edges together in one list as the value
graph = graph.map(lambda (x, y): (x, list(set(itertools.chain.from_iterable(list(y))))))
graph = graph.filter(lambda (x, y): len(y) != 0)

# Cache graph to speed up the searches
graph_cache = graph.cache()

print bfs(u'CAPTAIN AMERICA', graph_cache)
print bfs(u'MISS THING/MARY', graph_cache)
print bfs(u'ORWELL', graph_cache)