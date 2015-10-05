def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner


def flip_edge(elem):
		(node, neighbors) = elem
		vals = []
		for neighbor in neighbors:
			vals.append((neighbor, set([node])))
		return vals


# creates new adjacency list where it's only edges where each node is directed at other.
# using a hint given by professor on piazza @267
def get_double_sided_only(adj, num_partitions):
	reversed_adj = adj.flatMap(flip_edge).reduceByKey(lambda x, y: x.union(y), numPartitions=num_partitions).partitionBy(num_partitions)
	adj.partitionBy(num_partitions)
	new_adj = adj.join(reversed_adj).mapValues(lambda (adj_vals, reverse_adj_vals): set(adj_vals).intersection(reverse_adj_vals))
	return new_adj

def make_all_symettric(adj, num_partitions):
	reversed_adj = adj.flatMap(flip_edge).reduceByKey(lambda x, y: x.union(y), numPartitions=num_partitions).partitionBy(num_partitions)
	adj.partitionBy(num_partitions)
	#assert(copartitioned(reversed_adj, adj))
	new_adj = adj.join(reversed_adj).mapValues(lambda (adj_vals, reverse_adj_vals): set(adj_vals).union(reverse_adj_vals))
	return new_adj


def neighbors(elem):
	node, neighbors = elem
	min_val = min(neighbors)
	# Add all of the
	edges = [(min_val, set(neighbors))]
	for n in neighbors:
		edges.append(set([n]), min_val)
	return edges


#def updateLabel(tup):
#	label, neighbors = tup
#	min_neighbors = min(neighbors)
#	if min_neighbors < label:
#		return min(label, min_neighbors)
#	return min(label, min_neighbors)

def get_changed_ones(tup):
	(node, (old_labels, new_labels)) = tup
	old = list(old_labels)[0]
	if len(new_labels) == 0:
		return False
	else:
		return old > min(new_labels)

#def reducer1(tup):
# Basically we are going to uniquely identify the group nodes by the min value in each node.
# So, at each node, we are going to add 'edges' between 
def connected_components(adj, sc, numPartitions):
	accum1 = sc.accumulator(1)
	accum2 = sc.accumulator(1)

	adj = adj.partitionBy(numPartitions).cache()
	# Initialize the labels to themselves
	labels = adj.map(lambda (x, _): (x, x)).partitionBy(numPartitions).cache()

	while accum1.value != 0:
		accum1.value = 0
		#print adj.partitioner
		#print labels.partitioner
		assert(copartitioned(labels, adj))

		#labels = labels.join(adj).mapValues(updateLabel).partitionBy(numPartitions).cache()
		joined = labels.join(adj)
		#print joined.take(10)
		potential_new_labels = joined.values().flatMap(lambda (label, neighbors): [(n, label) for n in neighbors]).partitionBy(numPartitions).cache()
		assert(copartitioned(labels, potential_new_labels))
		unioned_labels = labels.cogroup(potential_new_labels)
		unioned_labels.filter(get_changed_ones).foreach(lambda x: accum1.add(1))

		labels = unioned_labels.mapValues(lambda (old_label, new_labels): min(list(old_label) + list(new_labels))).partitionBy(numPartitions).cache()
		#print potential_new_labels.take(100)
		#print unioned_labels.collect()
		#print labels.take(100)
		print 'Value: ', accum1.value
		#joined.mapValues(lambda (label, neighbors): neighbors).map(lambda x:)
		#adj.flatMap(lambda x: x). 
		#labels.foreach(lambda (node, label): accum1.add(label[1]))
		#print labels.take(10)
		#print accum1.value

		#accum1.value = 0
		#accum2.value = 0
		#adj.foreach(lambda _, accum1.add(1))
		#new_adj = adj.flatMap(neighbors).reduceByKey(lambda x, y: x | y)

		#new_adj.foreach(lambda _, accum1.add(1))
	#final_labels = labels.mapValues(lambda (lab, is_updated): lab)
	#print final_labels.take(10)
	return labels



# Comment out these lines when running on AWS
import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="Spark1")

import numpy as np 
import itertools

# make spark shut the hell up
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )


linklist = sc.textFile('../P5/links-simple-sorted.txt', 2)
titlelist = sc.textFile('../P5/titles_sorted.txt', 2)
#linklist = sc.textFile('../P5/generated_links_2mil_sorted.txt', 32)
#titlelist = sc.textFile('../P5/generated_titles_2mil_sorted.txt', 32)
#linklist = sc.textFile('../P5/generated_links_100k_sorted.txt', 32)
#titlelist = sc.textFile('../P5/generated_titles_100k_sorted.txt', 32)

#linklist = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
#titlelist = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

numerical_titles = titlelist.zipWithIndex().cache()

num_nodes = numerical_titles.count()
#num_Partitions = int(num_nodes/50)
num_Partitions = 256

# Borrowed from Professor's Github example on SparkPageRank
def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) - 1 for to in dests.split(' ')]
    return (int(src) - 1, dests)

split_list = linklist.map(link_string_to_KV).cache()

both_sided = get_double_sided_only(split_list, num_Partitions)
symmetrics = make_all_symettric(split_list, num_Partitions)

# Change these to be the appropriate start and end locations.
#start = "Kevin_Bacon"
#end = "Harvard_University"
#start = 'TITLE_8'
#end = "TITLE_4"

#start_node = numerical_titles.lookup(start)[0]
#end_node = numerical_titles.lookup(end)[0]
#print start_node, end_node

#dist = bfs(split_list, start_node, sc, num_Partitions, end_node)
#print distances.values().countByValue(), '\n'
#print dist.take(100)
##print "Distance to end node:", dist.map(lambda x: x).lookup(end_node)[0]

print '\n\n\n\n\n\n\n\n\n\n\nCalculating connected components....\n\n'
conn = connected_components(symmetrics, sc, num_Partitions)
#print "Number of componnents is:" 
grouped = conn.map(lambda (node, component): (component, node)).groupByKey().cache()
print 'Num components:'
print grouped.count(), '\n\n\n'
print "Number in largest component:"
largest_compoenent_number, elems_in_component = grouped.takeOrdered(1, key=lambda (x, y): (-1)*len(y))[0]
print len(elems_in_component)
print conn.values().countByValue(), '\n'


