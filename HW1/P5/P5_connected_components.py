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

def get_changed_ones(tup):
	(node, (old_labels, new_labels)) = tup
	old = list(old_labels)[0]
	if len(new_labels) == 0:
		return False
	else:
		return old > min(new_labels)

# Basically we are going to uniquely identify the group nodes by the min value in each node.
# The process will basicaly involve updating an RDD of labels for each node, which will be initialized to be 
# just itself. Then, each node will repeatedly get the labels of its neighbors if that label is lower, until there
# are no changes to the labels.
def connected_components(adj, sc, numPartitions):
	accum1 = sc.accumulator(1)
	accum2 = sc.accumulator(1)

	adj = adj.partitionBy(numPartitions).cache()
	# Initialize the labels to themselves
	labels = adj.map(lambda (x, _): (x, x)).partitionBy(numPartitions).cache()

	while accum1.value != 0:
		accum1.value = 0
		assert(copartitioned(labels, adj))

		joined = labels.join(adj)

		# Each node will give its labels to all of its neighbors
		potential_new_labels = joined.values().flatMap(lambda (label, neighbors): [(n, label) for n in neighbors]).partitionBy(numPartitions).cache()
		assert(copartitioned(labels, potential_new_labels))
		# Then, cogroup the labels that were given (potential new labels) with the old ones
		unioned_labels = labels.cogroup(potential_new_labels)
		# Count the number of changed labels by finding when the old label is not as good as one of the new ones
		unioned_labels.filter(get_changed_ones).foreach(lambda x: accum1.add(1))
		# pick the min of the old labels and any of the potential new labels
		labels = unioned_labels.mapValues(lambda (old_label, new_labels): min(list(old_label) + list(new_labels))).partitionBy(numPartitions).cache()
		print 'Value: ', accum1.value

	return labels

# Comment out these lines when running on AWS
import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="Spark1")

import numpy as np 
import itertools

# Reduce the amount that Spark logs to the console.
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

# These are local test files that I tested my code on. I got it to work for these!
linklist = sc.textFile('../P5/links-simple-sorted.txt', 2)
titlelist = sc.textFile('../P5/titles_sorted.txt', 2)
#linklist = sc.textFile('../P5/generated_links_2mil_sorted.txt', 32)
#titlelist = sc.textFile('../P5/generated_titles_2mil_sorted.txt', 32)
#linklist = sc.textFile('../P5/generated_links_100k_sorted.txt', 32)
#titlelist = sc.textFile('../P5/generated_titles_100k_sorted.txt', 32)

# Uncomment these to run on AWS.
#linklist = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
#titlelist = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

numerical_titles = titlelist.zipWithIndex().cache()
num_Partitions = 256

# Borrowed from Professor's Github example on SparkPageRank
def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) - 1 for to in dests.split(' ')]
    return (int(src) - 1, dests)

split_list = linklist.map(link_string_to_KV).cache()

# Extract the two types of graphs from the Wikipedia graph.
both_sided = get_double_sided_only(split_list, num_Partitions)
symmetrics = make_all_symettric(split_list, num_Partitions)

print '\n\n\n\n\n\n\n\n\n\n\nCalculating connected components....\n\n'
# Change symmetrics to both_sided to run this on the other graph extracted from the Wikipedia graph
conn = connected_components(symmetrics, sc, num_Partitions)
grouped = conn.map(lambda (node, component): (component, node)).groupByKey().cache()
print 'Num components:'
print grouped.count(), '\n\n\n'
print "Number in largest component:"
largest_compoenent_number, elems_in_component = grouped.takeOrdered(1, key=lambda (x, y): (-1)*len(y))[0]
print len(elems_in_component)


