from pyspark import SparkContext
from P5_bfs import *

sc = SparkContext("local", "Wikipedia graph")

# links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
links = sc.textFile('test.txt')
# page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
# page_names_indexed = page_names.zipWithIndex().map(lambda (n, i): (n, i + 1))

# parse and format into RDD
links_lists = links.map(lambda x: x.split(" "))
links_graph = links_lists.map(lambda l: (l[0][:len(l[0]) - 1], l[1:]))
links_graph = links_graph.map(lambda (n, l): (int(n), [int(x) for x in l]))

dist = rdd_bfs(1, links_graph)

print dist
