#######################################################################
### Problem 5 - Larger-scale Graph Processing with AWS & Spark[30%] ###
### P4_bfc.py														###
### Patrick Day 													###
### CS 205 HW1                                  					###
### Oct 4th, 2015													###
#######################################################################

########################
### Import Functions ###
########################
import pyspark
import numpy as np
import csv
from operator import add
import itertools
import string
import re

import os
sc = pyspark.SparkContext()
sc.setLogLevel('WARN')

########################
### Define Functions ###
########################
def copartitioned(RDD1, RDD2):
	"check if two RDDs are copartitioned"
	return RDD1.partitioner == RDD2.partitioner

def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)

def fun(i):
    return lambda x: (x[0], i + 1)

def find_distance(node, dist):
        for i in dist:
                if i[0] == node:
                        position = i[1]
        return position

def ss_bfs(page_ind, links_rdd, start_node, end_node):
	#TODO

# Import Sources
wiki_links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)#, use_unicode=False)
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)#, use_unicode=False)

# Add Index
page_ind = page_names.zipWithIndex().map(lambda (n, id): (n, id+1))

# Find Kevin Bacon! and Harvard
kevin_bacon = page_ind.filter(lambda (x,y): x == "Kevin_Bacon").collect()
kevin_bacon_name = unicode(kevin_bacon[0][1])
harvard_uni = page_ind.filter(lambda (x,y): x = "Harvard_University").collect()
harvard_uni_name = unicode(harvard_uni[0][1])

# Preprocess name with unicode
re_format = re.compile(r'[%s\s]+' % re.escape(string.punctuation))
links_split = wiki_links.(lambda x: re_format.split(x))
links_rdd = links_split.map(lambda x: (x[0], x[1:])).partitionBy(256).cache()

### Find Links from Kevin Bacon to Harvard ###
start_page = [(kevin_bacon_name, 0)]
end_page = harvard_uni_name
start_rdd = sc.parallelize(start_page).partitionBy(256)
depth = 0
stop_critieria = 4
while depth < stop_critieria:
    #assert copartitioned(full_rdd2, rdd)
    
    paren_rdd = start_rdd.filter(lambda x: x[1] == i).join(links_rdd).mapValues(lambda x: x[1])
    #assert copartitioned(rdd, rdd_parent)
    
    child_rdd = parent_rdd.flatMap(lambda x: x[1]).map(lambda x: (x, 1)).subtractByKey(start_rdd).distinct().partitionBy(256)
    #print 'distinct/subtract', i
    #assert copartitioned(rdd, rdd_children)
    
    next_child_rdd = child_rdd.map(fun(i), True)
    #print 'change radii', i
    #assert copartitioned(rdd, rdd_children)
    
    start_rdd = start_rdd.union(next_child_rdd).cache()
    #print 'union:', i
    #assert copartitioned(rdd, rdd_intermediate)

    i += 1 #iterator may not work on cluster
    print 'end of iter', i

### Display Kevin's information ###
kevin_list = start_rdd.collect()
pickle.dump(kevin_list, open("Kevin_Node_Graph", "wb"))

file = open("kevin_bacon_path.txt", "w")
	for line in kevin_list:
  		file.write("%s\n" % item)

# Initialize parameters
children_depth = 2
start_depth = 1
#child_node = harvard_uni_name

# Explore Parents of children
canidate_parent = rdd.filter(lambda x: x[1] == start_depth).partitionBy(256)
canidate_parent_of_child = canidate_parent.join(links_rdd).mapValues(lambda x: x[1])
found_parent = canidate_parent_of_child.filter(lambda x: harvard_uni_name in x[1]).keys()

# Collect Next Step
next_parent = found_parent.collect()
individual_parent = page_ind.filter(lambda (k,v): v ==int(next_parent[0])).collect()

all_poss_paths = ['Kevin_Bacon' individual_parent[0][0], 'Harvard_University']
print("Kevin's shortest Path:", all_poss_paths[:10])

pickle.dump(individual_parent, open("Kevin_to_Harvard.pkl", "wb"))

