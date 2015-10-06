from pyspark import SparkContext
from P5_bfs import *

# constants
NUM_PART = 64

sc = SparkContext("local", "Wikipedia graph")

# read in data
links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')

# index page names by 1
page_names_indexed = page_names.zipWithIndex().map(lambda (n, i): (n, i + 1))
page_names_indexed = page_names_indexed.cache()

# parse and format into RDD, cached to optimize
links_lists = links.map(lambda x: x.split(" "))
links_graph = links_lists.map(lambda l: (l[0][:len(l[0]) - 1], l[1:]))
links_graph = links_graph.map(lambda (n, l): (int(n), [int(x) for x in l]))
links_graph = links_graph.partitionBy(NUM_PART).cache()

# find Kevin_Bacon and Harvard_University
kev_idx = page_names_indexed.lookup("Kevin_Bacon")[0]           # 2729536
harv_idx = page_names_indexed.lookup("Harvard_University")[0]   # 2152782

# find prev dictionaries
prev_kev_harv = rdd_bfs(kev_idx, harv_idx, links_graph, sc)
prev_harv_kev = rdd_bfs(harv_idx, kev_idx, links_graph, sc)

# look backwards through prev dictionary to find the path
temp = harv_idx
path = [harv_idx]
while temp in prev_kev_harv:
    path.append(prev_kev_harv[temp])
    temp = prev_kev_harv[temp]
path.pop()

page_names_indexed = page_names_indexed.map(lambda (n, i): (i, n)).cache()

print "Path from Kevin Bacon to Harvard University:"
for p in path[::-1]:
    print page_names_indexed.lookup(p)[0]

temp = kev_idx
path = [kev_idx]
while temp in prev_harv_kev:
    path.append(prev_harv_kev[temp])
    temp = prev_harv_kev[temp]
path.pop()

print "Path from Harvard University to Kevin Bacon:"
for p in path[::-1]:
    print page_names_indexed.lookup(p)[0]
