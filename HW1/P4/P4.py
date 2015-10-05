from pyspark import SparkContext
from P4_bfs import *

sc = SparkContext("local", "Marvel charater graph")
accum = sc.accumulator(0)

marvel_csv = sc.textFile("source.csv")

# parse and format into RDD
marvel_rdd = marvel_csv.map(lambda l: l.split('"'))
marvel_rdd = marvel_rdd.map(lambda (a, b, c, d, e): (d, b))

# group characters in comics
comic_dict = marvel_rdd.map(lambda (k, v): (k, [v]))
comic_dict = comic_dict.reduceByKey(lambda x, y: x + y)

# represent as adjacency list
marvel_graph = comic_dict.map(lambda (k, l):
                                [(c, [x for x in l if x != c]) for c in l])
marvel_graph = marvel_graph.flatMap(lambda l: l)
marvel_graph = marvel_graph.reduceByKey(lambda x, y: x + y)
marvel_graph = marvel_graph.map(lambda (k, v): (k, list(set(v))))

# find distance dictionaries for some characters
cap_dist = rdd_bfs("CAPTAIN AMERICA", marvel_graph)
mary_dist = rdd_bfs("MISS THING/MARY", marvel_graph)
orwl_dist = rdd_bfs("ORWELL", marvel_graph)

# print number of nodes reachable from each character, within 10 steps
print len(cap_dist)
print len(mary_dist)
print len(orwl_dist)
