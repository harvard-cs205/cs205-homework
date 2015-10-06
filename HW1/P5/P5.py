import numpy as np
import pyspark
sc = pyspark.SparkContext(appName="P4")

from P5_bfs import find_all_shortest_paths
from P5_connected_components import connected_components

sc.setLogLevel('WARN')

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
titles = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')

titles = titles.zipWithIndex().map(lambda (k,v): (v + 1,k))

titles_rev = titles.map(lambda (k,v): (v, k))

links = links.map(lambda line: (int(line.split(": ")[0]), map(int, line.split(": ")[1].split())))
full_graph = links.flatMap(lambda (k, v): [(k, vl) for vl in v])

def compute_shortest_paths_and_map(hu_num, kb_num, full_graph, sc):
  ret = find_all_shortest_paths(hu_num, kb_num, full_graph, sc)
  raw_nodes = ret.flatMap(lambda x: x).map(lambda x: (x, x))
  relevant_dict = raw_nodes.join(titles)
  rdmap = relevant_dict.map(lambda (k, v): (k, v[1])).collectAsMap()
  print ret.map(lambda x: [rdmap[a] for a in x]).collect()

kb_num = titles_rev.lookup('Kevin_Bacon')[0]
hu_num = titles_rev.lookup('Harvard_University')[0]
compute_shortest_paths_and_map(hu_num, kb_num, full_graph)
compute_shortest_paths_and_map(kb_num, hu_num, full_graph)