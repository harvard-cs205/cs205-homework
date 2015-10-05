import findspark
findspark.init()
import pyspark
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("app").setMaster("local")
sc = SparkContext(conf=conf)

links = sc.textFile('links.txt')

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

links_RDD = links.map(links_mapper)
asym_links_RDD, sym_links_RDD = pre_process(links_RDD)


def split_tuples_mapper((k, v)):
    res = list()
    for item in v:
        res.append((k, item))
    return res

def hash_neighbors_reducer(neighbors1, neighbors2):
    for item in neighbors2:
        if item not in neighbors1:
            neighbors1.append(item)
    return neighbors1

def possible_hash_mapper((k, v)):
    res = list()
    for item in v:
        res.append((item, k))
    res.append((k, k)) # add itself
    return res
	
def get_cc_by_hash_to_min(_index_neighbors_rdd):
	index_hash_rdd = _index_neighbors_rdd.map(lambda (k, v): (k, k))
	old_number_of_clusters = -1

	while True:
		hash_neighbors_rdd = index_hash_rdd.join(_index_neighbors_rdd).map(lambda (k, (h, neighbors)): (h, neighbors)).reduceByKey(hash_neighbors_reducer)

		number_of_clusters = hash_neighbors_rdd.count()
		if old_number_of_clusters == number_of_clusters:
			break
		old_number_of_clusters = number_of_clusters

		hash_index_rdd = index_hash_rdd.map(lambda (k, v): (v, k))

		index_possible_min_hash_rdd = hash_neighbors_rdd.flatMap(possible_hash_mapper).reduceByKey(lambda pos_hash1, pos_hash2: pos_hash1 if pos_hash1 < pos_hash2 else pos_hash2)
		old_new_hash_rdd = index_possible_min_hash_rdd.join(index_hash_rdd).map(lambda (k, (new_hash, old_hash)): (old_hash, new_hash if old_hash > new_hash else old_hash)).reduceByKey(lambda new_hash1, new_hash2: new_hash1 if new_hash1 < new_hash2 else new_hash2)
		index_hash_rdd = hash_index_rdd.join(old_new_hash_rdd).map(lambda (k, (index, new_hash)): (index, new_hash))
	return index_hash_rdd

sym_result_cluster_index_rdd = get_cc_by_hash_to_min(sym_links_RDD).map(lambda (k, c): (c, k)).groupByKey().mapValues(lambda x: list(x))
asym_result_cluster_index_rdd = get_cc_by_hash_to_min(asym_links_RDD).map(lambda (k, c): (c, k)).groupByKey().mapValues(lambda x: list(x))
print sym_result_cluster_index_rdd.collect(), asym_result_cluster_index_rdd.collect()