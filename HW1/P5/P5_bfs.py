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

page_names = page_names.zipWithIndex().map(lambda (n, id): (n, id + 1))

Kevin_Bacon = page_names.filter(lambda (K, V): K == 'Kevin_Bacon').collect()
assert len(Kevin_Bacon) == 1
Kevin_Bacon = unicode(Kevin_Bacon[0][1])

Harvard_University = page_names.filter(lambda (K, V): K == 'Harvard_University').collect()
assert len(Harvard_University) == 1
Harvard_University = unicode(Harvard_University[0][1])

regex = re.compile(r'[%s\s]+' % re.escape(string.punctuation))
split_links = links.map(lambda s: regex.split(s))

full_rdd2 = split_links.map(lambda x: (x[0], x[1:])).partitionBy(256).cache()


############################################# Kevin to Harvard #############################

start_node = Kevin_Bacon
end_node = Harvard_University

start = [(start_node, 0)] 
rdd = sc.parallelize(start).partitionBy(256)
i = 0

num_runs = 4

while i < num_runs:
    assert copartitioned(full_rdd2, rdd)
    
    rdd_parent = rdd.filter(lambda x: x[1] == i).join(full_rdd2).mapValues(lambda x: x[1])
    assert copartitioned(rdd, rdd_parent)
    
    rdd_children = rdd_parent.flatMap(lambda x: x[1]).map(lambda x: (x, 1)).subtractByKey(rdd).distinct().partitionBy(256)
    print 'distinct/subtract', i
    assert copartitioned(rdd, rdd_children)
    
    rdd_intermediate = rdd_children.map(fun(i), True)
    print 'change radii', i
    assert copartitioned(rdd, rdd_children)
    
    rdd = rdd.union(rdd_intermediate).cache()
    print 'union:', i
    assert copartitioned(rdd, rdd_intermediate)

    i += 1 #iterator may not work on cluster
    print 'end of iter', i


#Write Out Information

#Write Out Node Information for complete graph
node_distances = rdd.collect()
pickle.dump(node_distances, open( "KB_Harvard_Distances.pkl", "wb" ) )


#Establish know initial parameters
child_distance = 2
filter_distance = child_distance - 1
child = Harvard_University


# We know from runnin BFS-SS on the wiki set with Kevin Bacon as the root node, that Harvard University if distance 2
# so we just need to back off Harvard and find out who the parents are, to fill in the shortest path. There will
# obviously be more than on shortest path between the two nodes. 
potential_parents = rdd.filter(lambda x: x[1] == filter_distance).partitionBy(256) #667

potential_parents_with_children = potential_parents.join(full_rdd2).mapValues(lambda x: x[1]) #667, p

parents_of_child = potential_parents_with_children.filter(lambda x: child in x[1]).keys() #364, pp

#Write out middle nodes
middle_step = parents_of_child.collect()

single_parent = page_names.filter(lambda (K, V): V == int(middle_step[0])).collect()

 

all_steps = ['Kevin_Bacon', single_parent[0][0], 'Harvard_University']

print "KB Shortest Path:", all_steps

pickle.dump(single_parent, open( "KB_HARVARD.pkl", "wb" ) )




############################################# Harvard to Kevin #############################

start_node = Harvard_University
end_node = Kevin_Bacon

start = [(start_node, 0)] 
rdd = sc.parallelize(start).partitionBy(256)
i = 0

num_runs = 4

while i < num_runs:
    assert copartitioned(full_rdd2, rdd)
    
    rdd_parent = rdd.filter(lambda x: x[1] == i).join(full_rdd2).mapValues(lambda x: x[1])
    assert copartitioned(rdd, rdd_parent)
    
    rdd_children = rdd_parent.flatMap(lambda x: x[1]).map(lambda x: (x, 1)).subtractByKey(rdd).distinct().partitionBy(256)
    print 'distinct/subtract', i
    assert copartitioned(rdd, rdd_children)
    
    rdd_intermediate = rdd_children.map(fun(i), True)
    print 'change radii', i
    assert copartitioned(rdd, rdd_children)
    
    rdd = rdd.union(rdd_intermediate).cache()
    print 'union:', i
    assert copartitioned(rdd, rdd_intermediate)

    i += 1 #iterator may not work on cluster
    print 'end of iter', i

#Write Out Information

#Write Out Node Information for complete graph
#node_distances = rdd.collect()
#pickle.dump(node_distances, open( "Harvard_Distances_KB.pkl", "wb" ) )


#Establish know initial parameters
child_distance = 3
filter_distance = child_distance - 1
child = Kevin_Bacon


potential_parents = rdd.filter(lambda x: x[1] == filter_distance).partitionBy(256) 

potential_parents_with_children = potential_parents.join(full_rdd2).mapValues(lambda x: x[1]) 

parents_of_child = potential_parents_with_children.filter(lambda x: child in x[1]).keys() 

#Write out middle nodes
middle_step = parents_of_child.collect()

one_behind_harvard = page_names.filter(lambda (K, V): V == int(middle_step[1])).collect()

#print 'harvard middle', middle_step
#print 'one behind Harvard', one_behind_harvard[0][0]

filter_distance -= 1
child = unicode(one_behind_harvard[0][1])

potential_parents = rdd.filter(lambda x: x[1] == filter_distance).partitionBy(256) 

potential_parents_with_children = potential_parents.join(full_rdd2).mapValues(lambda x: x[1]) 

parents_of_child = potential_parents_with_children.filter(lambda x: child in x[1]).keys() 

second_step = parents_of_child.collect()

two_behind_harvard = page_names.filter(lambda (K, V): V == int(second_step[0])).collect()


#print 'harvard second', middle_step
#print 'one behind Harvard', two_behind_harvard[0][0]

both_steps = ['Harvard_University', two_behind_harvard[0][0], one_behind_harvard[0][0], 'Kevin_Bacon']

print "Harvard Shortest Path:", both_steps 

pickle.dump(both_steps, open( "KB_HARVARD.pkl", "wb" ) )
















