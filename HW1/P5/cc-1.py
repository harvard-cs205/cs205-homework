import findspark
findspark.init()
import pyspark
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("app").setMaster("local")
sc = SparkContext(conf=conf)

links = sc.textFile('links.txt').cache()
page_names = sc.textFile('titles-sorted.txt').cache()


def expand_values((k, v)):
	res = list()
	for i in v:
		res.append((k, i))
	return res

def expand_values_union((k, v)):
	res = list()
	for i in v:
		res.append((i, k))
	return res

def pre_process(links_rdd):
	rdd = links_rdd
	rdd1 = rdd.flatMap(expand_values)
	rdd2 = rdd.flatMap(expand_values_union)
	rdd_intersect = rdd1.intersection(rdd2).groupByKey()
	rdd_union = rdd1.union(rdd2).groupByKey()
	return rdd_intersect, rdd_union

def links_mapper(w):
	if ': ' not in w:
		return -1, -1
	w = w.split(': ')
	index = int(w[0])
	neighbors = []
	for item in w[1].split(' '):
		if len(item) > 0:
			neighbors.append(int(item))
	return index, neighbors

max_distance = 10000000
max_cid = -1


page_index_RDD = page_names.zipWithIndex().map(lambda (k, v): (k, v+1))
links_RDD = links.map(links_mapper).cache()

links_intersect, links_union = pre_process(links_RDD)

# TODO: rrun case for union !!
graph_RDD = links_intersect.map(lambda (k, v): (k, (False, v, max_distance, max_cid)))

def bfs_one_node(origin_index):
	global sc, graph_RDD, max_distance, max_cid

	# construct graphRDD
	def construct_mapper((key, (flag, neighbors, distance, cid))):
		if key == origin_index:
			return key, (True, neighbors, 0, key)
		return key, (False, neighbors, max_distance, cid)

	pre_graph_rdd = graph_RDD.map(construct_mapper)
	present_rdd = pre_graph_rdd

	def mapper((key, (flag, neighbors, distance, cid))):
		result = list()
		result.append((key, (False, neighbors, distance, cid)))
		if distance == turn_id:
			for otherKey in neighbors:
				result.append((otherKey, (True, [], distance+1, cid)))
		return result

	def reducer((flag1, adj1, dist1, cid1), (flag2, adj2, dist2, cid2)):
		adj = adj1 if len(adj1) > len(adj2) else adj2
		cid = cid1 if cid1 != max_cid else cid2
		if dist1 < dist2:
			return flag1, adj, dist1, cid
		else:
			return flag2, adj, dist2, cid

	turn_id = 0
	while True:
		last_num = present_rdd.filter(lambda (k, (f, a, d, c)): c == max_cid).count()
		present_rdd = present_rdd.flatMap(mapper).reduceByKey(reducer)
		present_num = present_rdd.filter(lambda (k, (f, a, d, c)): c == max_cid).count()
		if present_num == last_num:
			break
		else:
			turn_id += 1
		print turn_id
	return present_rdd

 # cc by bfs
next_term = 1
kk = 0
while True:
	kk+=1
	print kk,'kkk'
	graph_RDD = bfs_one_node(next_term)
	new_num = graph_RDD.filter(lambda (k, (f, a, d, c)): c == max_cid).count()
	print new_num
	print graph_RDD.filter(lambda (k, (f, a, d, c)): c == max_cid).collect()
	if new_num != None:
			next_term = graph_RDD.filter(lambda (k, (f, a, d, c)): c == max_cid).take(1)
	else:
		break

index_cluster_RDD = graph_RDD.map(lambda (k, (f, a, d, c)): (k, c))
print index_cluster_RDD.collect()