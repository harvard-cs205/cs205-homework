#import findspark
#findspark.init()
#import pyspark
import time
start = time.time()
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("app").setMaster("local")
sc = SparkContext(conf=conf)

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
#links = sc.textFile('links.txt')
start = time.time()

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

def reverse_edges(node):
    edges = []
    for neighbor in node[1]:
        edges.append((neighbor, set([node[0]])))
    return edges


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

def dataStruCons((k, v)):
    return (k, (False, v, max_distance, max_cid))

max_distance = 10000000
max_cid = -1

links_RDD = links.map(links_mapper).partitionBy(4).cache()
links_intersect, links_union = pre_process(links_RDD)

sym_graph_RDD = links_union.map(dataStruCons).cache()
revRDD = links_RDD.flatMap(reverse_edges).reduceByKey(lambda x,y: x.union(y)).partitionBy(4)
asym_graph_RDD = links_RDD.join(revRDD).map(lambda (k,v): (k, (False, (set(v[0]).intersection(v[1])),max_distance,max_cid)))

def bfs_one_node(origin_index, _rdd):
    global sc, max_distance, max_cid

    # construct graphRDD
    def construct_mapper((key, (flag, neighbors, distance, cid))):
        if key == origin_index:
            return key, (True, neighbors, 0, key)
        return key, (False, neighbors, max_distance, cid)

    pre_graph_rdd = _rdd.map(construct_mapper)
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
    return present_rdd

 # cc by bfs
def get_cc(rdd):
    next_item = 1
    while True:
            rdd = bfs_one_node(next_item, rdd)
            new_num = rdd.filter(lambda (k, (f, a, d, c)): c == max_cid).count()
            if new_num != 0:
                    next_item = rdd.filter(lambda (k, (f, a, d, c)): c == max_cid).take(1)[0][0]
            else:
                    break
    return rdd


sym_result_cluster_index_rdd = get_cc(sym_graph_RDD).map(lambda (k, (f, a, d, c)): (c, k)).groupByKey().mapValues(lambda x: list(x))
asym_result_cluster_index_rdd = get_cc(asym_graph_RDD).map(lambda (k, (f, a, d, c)): (c, k)).groupByKey().mapValues(lambda x: list(x))
ccLogFile = open('P5CCLogFile.txt', 'w')

ccLogFile.write('Symmetric Graph: ' + '# of Connected Components is : ' 
                 + str(sym_result_cluster_index_rdd.count()) + 
                 '\tMax # of Elements is : ' + str(len(sym_result_cluster_index_rdd.takeOrdered(1,lambda x: -len(x[1]))[0][1]))
                 + '\nAsymmetric Graph: ' + '# of Connected Components is: '
                 + str(asym_result_cluster_index_rdd.count()) + 
                 '\tMax # of Elements is : ' + str(len(asym_result_cluster_index_rdd.takeOrdered(1,lambda x: -len(x[1]))[0][1])) + '\n'
                 )
ccLogFile.close()
print time.time()-start
                


