

import findspark
findspark.init()
import pickle
import string
import re
import urllib2
import csv
import itertools
import math as ma
import os
import pyspark


sc = pyspark.SparkContext()
#sc.setLogLevel('WARN')




class bfs:

    
    def __init__(self, start_node = 'CAPTAIN AMERICA', number_interations = 10, partition = 16):

    	#self.graph = graph

    	self.start_node = start_node

    	self.number_interations = number_interations

        self.partition = partition


    def generate_marvel(self):
        url = 'http://exposedata.com/marvel/data/source.csv'
        response = urllib2.urlopen(url)
        cr = csv.reader(response)
        char = []
        for row in cr:
            char.append(row)

        partition = 16


        char_rdd = sc.parallelize(char).repartition(partition)
        char_rdd_kv = char_rdd.map(lambda x: (x[1], x[0]))
        char_group_rdd = char_rdd_kv.groupByKey()
        char_list = char_group_rdd.mapValues(list).collect()
        char_list_1 = sc.parallelize(char_list).repartition(partition)
        char_rdd_2 = char_list_1.flatMap(lambda x: list(itertools.combinations(x[1], 2)))
        char_rdd_reverse = char_rdd_2.map(lambda x: (x[1], x[0]))
        union = char_rdd_2.union(char_rdd_reverse)
        union_2 = union.groupByKey().mapValues(list)
        union_3 = union_2.map(lambda x: (x[0], list(set(x[1]))))   
        
        return union_3

        


    def copartitioned(self, RDD1, RDD2):

        "check if two RDDs are copartitioned"

        return RDD1.partitioner == RDD2.partitioner

    def fun(self, i):
    	
    	return lambda x: (x[0], i + 1)



    def bfs_search(self):
        
        full_rdd2 = self.generate_marvel().partitionBy(self.partition)
        accum = sc.accumulator(0)
        accum_last = 1
        rdd = sc.parallelize([(self.start_node, 0)]).partitionBy(self.partition)
        i = 0
        
        while accum_last != accum.value:

		    accum_last = accum.value
		    print 'Last iteration accum:', accum_last
		    assert self.copartitioned(full_rdd2, rdd)
		    
		    rdd_parent = rdd.filter(lambda x: x[1] == i).join(full_rdd2).mapValues(lambda x: x[1])
		    assert self.copartitioned(rdd, rdd_parent)

		    rdd_parent.foreach(lambda x: accum.add(1))
		    print 'This iteration accum:', accum.value
		    
		    rdd_children = rdd_parent.flatMap(lambda x: x[1]).map(lambda x: (x, 1)).subtractByKey(rdd).distinct().partitionBy(self.partition)
		    print 'distinct/subtract', i
		    assert self.copartitioned(rdd, rdd_children)
		    
		    rdd_intermediate = rdd_children.map(self.fun(i), True)
		    print 'change radii', i
		    assert self.copartitioned(rdd, rdd_children)
		    
		    rdd = rdd.union(rdd_intermediate).cache()
		    print 'union:', i
		    assert self.copartitioned(rdd, rdd_intermediate)

		    #iterator may not work on cluster
		    print 'end of iter', i
		    i += 1 

        out = rdd.collect()
        return out



