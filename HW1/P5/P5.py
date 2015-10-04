from P5_bfs import *
from P5_connected_components import *
from pyspark import SparkContext, SparkConf

#Setup
conf = SparkConf().setAppName("wikipedia_graph")
sc = SparkContext(conf=conf)

def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

def cleanup(line):
    splitted = line.split(' ')
    return (splitted.pop(0).strip(':'), splitted)

def number_to_name(num, kvs):
    name_entry = kvs.filter(lambda KV: KV[1] == num)
    assert name_entry.count() == 1
    return name_entry.first()[0]
    
def name_to_number(name, kvs):
    number = kvs.lookup(name)
    assert len(number) == 1
    return number[0]
    
def symmetric_links_union(graph):
    inverted_pairs = graph.flatMapValues(lambda v: v).map(lambda KV: (KV[1], [KV[0]])).partitionBy(24)
    assert copartitioned(graph, inverted_pairs)
    full_graph = graph.union(inverted_pairs).reduceByKey(lambda x, y: x + y).mapValues(lambda v: frozenset(v))
    return full_graph

def symmetric_links_intersection(graph):
    keys = graph.mapValues(lambda _: [])
    pairs = graph.flatMapValues(lambda v: v)
    inverted_pairs = pairs.map(lambda KV: (KV[1], KV[0])).partitionBy(24)
    assert copartitioned(pairs, inverted_pairs)
    intersect = pairs.intersection(inverted_pairs).mapValues(lambda v:[v]).partitionBy(24)
    assert copartitioned(intersect, keys)
    full_graph = intersect.union(keys).reduceByKey(lambda x, y: x + y).mapValues(lambda v: frozenset(v))
    return full_graph
#Setup
links_lines = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
links_intermediate = links_lines.map(cleanup).map(lambda KV: (int(KV[0]), [int(v) for v in KV[1]])).partitionBy(24).cache()
links = links_intermediate.mapValues(lambda v: frozenset(v)).cache()
name_lookup = page_names.zipWithIndex().mapValues(lambda v: v+1).sortByKey().cache()

#Shortest Path
kevin_bacon = name_to_number('Kevin_Bacon', name_lookup)
harvard_university = name_to_number('Harvard_University', name_lookup)
kh_bfs_result = bfs_parents(links, kevin_bacon, harvard_university)
kh_path = shortest_path(kh_bfs_result, harvard_university)
kh_named_path = [(number_to_name(node, name_lookup), distance) for (node, distance) in kh_path]
hk_bfs_result = bfs_parents(links, harvard_university, kevin_bacon)
hk_path = shortest_path(hk_bfs_result, kevin_bacon)
hk_named_path = [(number_to_name(node, name_lookup), distance) for (node, distance) in hk_path]

#Connected Components
links_union = symmetric_links_union(links_intermediate).partitionBy(16).cache()
union = connected_components(links_union)
union_data = (len(union), max(union))
links_intersection = symmetric_links_intersection(links_intermediate).partitionBy(16).cache()
intersection = connected_components(links_intersection)
intersection_data = (len(intersection), max(intersection))

print (kh_named_path, hk_named_path, union_data, intersection_data)
