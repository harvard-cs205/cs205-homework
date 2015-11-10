import pickle
import string
import re
import urllib2
import itertools
import math as ma
import os
import pyspark

sc = pyspark.SparkContext()
sc.setLogLevel('WARN')

def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)


def fun(i):
    return lambda x: (x[0], i + 1)


def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner


def find_distance(node, dist):
        for i in dist:
                if i[0] == node:
                        position = i[1]
        return position


links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

regex = re.compile(r'[%s\s]+' % re.escape(string.punctuation))
split_links = links.map(lambda s: regex.split(s))

paritition = 256

#get initial sample
sample = split_links.flatMap(lambda x: x[0]).sample(True, 1, None)
start_node = random.choice(sample.collect())
print start_node
start = [(start_node, 0)]


rdd = sc.parallelize(start).partitionBy(partition)
#graph_1 = graph_1.partitionBy(partition).cache()
full_rdd2 = graph_1.partitionBy(partition).cache()
accum_2 = sc.accumulator(0)
accum_last_2 = 100

while accum_last_2 != accum_2.value:

    accum = sc.accumulator(0)
    accum_last = 1

    #while i < num_runs:
    while accum_last != accum.value:
        accum_last = accum.value
        print 'Last iteration accum:', accum_last
        assert copartitioned(full_rdd2, rdd)

        rdd_parent = rdd.filter(lambda x: x[1] == i).join(full_rdd2).mapValues(lambda x: x[1])
        assert copartitioned(rdd, rdd_parent)

        rdd_parent.foreach(lambda x: accum.add(1))
        print 'This iteration accum:', accum.value

        rdd_children = rdd_parent.flatMap(lambda x: x[1]).map(lambda x: (x, 1)).subtractByKey(rdd).distinct().partitionBy(partition)
        print 'distinct/subtract', i
        assert copartitioned(rdd, rdd_children)

        rdd_intermediate = rdd_children.map(fun(i), True)
        print 'change radii', i
        assert copartitioned(rdd, rdd_children)

        rdd = rdd.union(rdd_intermediate).cache()
        print 'union:', i
        assert copartitioned(rdd, rdd_intermediate)

         #iterator may not work on cluster
        print 'end of iter', i
        i += 1
    
    accum_last_2 = accum_2.value
    print 'Outer Last iteration accum:', accum_last_2
    
    #subtrace the component
    full_rdd2 = full_rdd2.subtractByKey(rdd).partitionBy(partition)
    full_rdd2.foreach(lambda x: accum.add(1))
    print 'Outer iteration accum:', accum.value
        
    sample = full_rdd2.flatMap(lambda x: x[0]).sample(True, 1, None)
    start_node = random.choice(sample.collect())
    print 'New start:', start_node
    
    start = [(start_node, 0)] 
    rdd = sc.parallelize(start).partitionBy(partition)


cc = rdd.collect()
pickle.dump(cc, open( "connected_components.pkl", "wb" ) )
