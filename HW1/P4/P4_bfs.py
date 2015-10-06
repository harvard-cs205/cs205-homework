###################################################
### Problem 4 - Graph processing in Spark [25%] ###
### P4_bfc.py									###
### Patrick Day 								###
### CS 205 HW1                                  ###
### Oct 4th, 2015								###
###################################################

########################
### Import Functions ###
########################
import pyspark
import numpy as np
import csv
from operator import add
import itertools

import os
sc = pyspark.SparkContext()
sc.setLogLevel('WARN')

########################
### Define Functions ###
########################
def copartitioned(RDD1, RDD2):
	"check if two RDDs are copartitioned"
	return RDD1.partitioner == RDD2.partitioner

def build_graph(lines):
	### Build the hero graph ###
	hero_rdd = sc.parallelize(lines).partitionBy(10)
	comic_rdd = hero_rdd.map(lambda x: (x[1], x[0]))#, preservesPartitioning=True)
	
	# Group all comics together by key
	comic_list = comic_rdd.groupByKey().mapValues(list).collect()
	comic_flat_rdd = sc.parallelize(comic_list).partitionBy(10)
	
	# Create hero pairs
	hero_pairs = comic_flat_rdd.flatMap(lambda x: list(itertools.combinations(x[1], 2)))
	hero_pair_rev = hero_pairs.map(lambda x: (x[1], x[0]), preservesPartitioning=True)
	
	# Bring all pairs back together and group
	all_heros = hero_pairs.union(hero_pair_rev)
	hero_combine = all_heros.groupByKey().mapValues(list)

	# Preprocess the graph according to (k=Hero, (v0=depth, v1=seen/processed/not_see))
	ref_graph = hero_combine.map(lambda x: (x[0], list(set(x[1]))), preservesPartitioning=True)
	graph_init = ref_graph.map(lambda x: (x[0], (x[1], 100, 0))).partitionBy(10).cache()

	return graph_init

def ss_bfs(graph_init, hero, partition=10):
	
	### Initialize first iteration ###
	
	# Create Hero RDD and set parent to processed
	hero_values = graph_init.filter(lambda x: x[0] == hero)
	processed_parent = hero_values.map(lambda x: (x[0], (0, 2)), preservesPartitioning=True)
	parent = hero_values.map(lambda x: (x[0][0], (x[1][0], 0, 1)), preservesPartitioning=True)
	
	# Update parent, find neighbors, and put together
	nei_init_rdd = parent.flatMap(lambda x: x[1][0]).map(lambda x: (x,(1,1)), preservesPartitioning=True).subtractByKey(processed_parent).partitionBy(partition)
	hero_graph = nei_init_rdd.union(processed_parent).partitionBy(partition).cache()
	
	# Initialize accumlators and depth after preprocessing
	accum_par = sc.accumulator(0)
	accum_nei = sc.accumulator(0)
	accum_update = 1
	depth = 2

	### SS_BDS ###
	while(accum_update!=accum_par.value):
	    
	    # Get heros to process from graph, update parent accumlator
	    parent_rdd = hero_graph.filter(lambda x: x[1][-1] == 1).join(graph_init).map(lambda x: (x[0], x[1][1]), preservesPartitioning=True)
	    parent_count = parent_rdd.count()

	    # Increment Accumlate passed on the number of parents processed
	    accum_update = accum_par.value
	    parent_rdd.foreach(lambda x: accum_par.add(1))
	  
	  	# Print Depth, Parents and Children
	    print ('Graph Depth:', depth)
	    print ("Childred currently being processed (k,v,v=1):", parent_count)
	    print ("Parents that have been processed (k,v,v=2):", accum_par.value)
	    
	    # Turn Neighbors to process and update neighbor accumlator
	    process_nei = parent_rdd.flatMap(lambda x: x[1][0]).map(lambda x: (x, (depth,1)), preservesPartitioning=True).distinct().subtractByKey(hero_graph).partitionBy(partition)
	    process_nei.foreach(lambda x: accum_nei.add(1))
	    
	    # Check if Parent and Children are copartitioned
	    print("Check Parent, Children, and processed Children partitions with Hero Graph")
	    assert copartitioned(parent_rdd, hero_graph)
	    assert copartitioned(process_nei, hero_graph)
	    

	    # Update Processed to Seen (Seen = 2)
	    seen_nei = hero_graph.filter(lambda x: x[1][-1] == 1).map(lambda x: (x[0], (x[1][0], 2)), preservesPartitioning=True)
	    assert copartitioned(seen_nei, hero_graph)
	    print("All Passed!")
	    
	    # Update the Hero graph
	    hero_graph = hero_graph.filter(lambda x: x[1][-1] == 2).union(seen_nei).union(process_nei).cache()
	    print()
			
	    depth+=1
	    
	# Return graph and Max depth of graph
	return(hero_graph, list(hero_graph.max(key = lambda x: x[1][0]))[1][0])
