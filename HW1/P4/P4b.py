import findspark
findspark.init()
import pyspark
import csv
import numpy as np 


sc = pyspark.SparkContext()
source_reader = csv.reader(open("source.csv", 'rb'), delimiter=',')

# readuce by key to get RDD with issue -> [super_hero list]
issue_sh = sc.parallelize(list(source_reader), 100).map(lambda x: (x[1].strip(), [x[0].strip()])).reduceByKey(lambda x, y: x + y)

def construct_edges(x):
    i, s_list = x
    l = len(s_list)
    edges = []
    for i in range(l):
        for j in range(l):
            if i != j: 
                edges.append(((s_list[i], s_list[j]), None))
    return edges

edges = issue_sh.flatMap(construct_edges).distinct().map(lambda x: (x[0][0], [x[0][1]]))

# need to define a source to initailize v_dist with
source = "CAPTAIN AMERICA"
# might want to broadcast this to all of the partitions 
def map_adj_list(x):
    if x[0] == source:
        return (x[0], (False, x[1])) 
    else:
        return (x[0], (False, x[1]))
adj_list = edges.reduceByKey(lambda x, y: x + y).map(map_adj_list)

# so we will keep a v_dist_neighs that we will reduce at the very end? 
v_dist_neighs = adj_list.filter(lambda x: x[0] == source).map(lambda x: (x[0], (0, False, x[1][1]))) 

def closure():
    acc = sc.accumulator(0)
    def map_vdist(x):
        k, v = x
        d, explored_flag, n_list = v
        if explored_flag: 
            return [(k, (d, explored_flag))]
        else:
            acc.add(1)
            v_list = [(k, (d, True))]
            for n in n_list:
                v_list.append((n, (d + 1, False)))
            return v_list 
    return acc, map_vdist

def closure2():
    acc = sc.accumulator(0)
    def map_vdist(x):
        k, v = x
        explored_flag, n_list = v
        if not explored_flag: 
            acc.add(1)
            v_list = []
            for n in n_list:
                v_list.append((n, (d + 1, False)))
            return v_list 
    return acc, map_vdist

def closure3(dist):
    acc = sc.accumulator(0)
    # this will return the new adj_list with extra elements
    # they will have flag False 
    def map_adjlist(x):
        k, v = x
        explored_flag, n_list = v
        if explored_flag:
            return [(k, v)]
        else:
            acc.add(1)
            v_list = [(k, (True, n_list))]
            for n in n_list:
                # don't even need to add dist because we can just do it after 
                v_list.append((n, (dist, False, [])))
# what if we added neighbors to v_list, did the flatMap over v_list

# need a new map_v_dist that 
# whats the problem: th
# we dont know when we are mapping over the adjacency list if the current element is supposed to be explored or not if its flag is False(actually we need it 

acc = sc.accumulator(1)
dist = 1
while(acc.value > 0):
    # might need to make this more efficient by combining adj_list and v_dist_neighs and filtering before shuffle operations 
    acc, map_adjlist = closure3()
    #acc, map_vdist = closure2()
    # filter out the edges from the adj list that have not been touched before and run flatMap 
    # do this by filtering out every node that has np.inf as distance or has been touched (flag is True)
    # then expand the nodes that should be explored right now
    # we first want the new adj_list
    adj_list_new_vs = adj_list.flatMap(map_adjlist)
      
    adj_list = adj_list_new_vs.filter(lambda x: x[1][0])

    new_vs = adj_list.filter(lambda x: (not x[1][1]) and (x[1][0] < np.inf)).flatMap(map_vdist)

    # other way to do this is to include the old nodes and reduce by key to get only new nodes for join
    # we can filter out the elements of adj_list that have already been touched before we do the join
    # might want to cache this from above 
    new_v_neighs = new_vs.join(adj_list.filter(lambda x: not x[1][1])).map(lambda x: (x[0], 
                                                                            (x[1][0][0], x[1][0][1],
                                                                             x[1][1][2])))
    # or who have a smaller distance than the current distance? 
    new_v_dist_neighs = new_vs.join(adj_list).map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][1])))
    # map to get the right distances, flag values 
    
    # don't actually need the neighbors, just node, distance 
    v_dist_neighs = v_dist_neighs.union(new_v_dist_neighs)
    dist += 1
# do a reduceByKey on the v_dist_neighs, take the minimum distance for each node 
#v_dist = v_dist.reduceByKey(lambda x, y: min(x, y))

print v_dist_neighs.count()
print dist 
print acc.value
#print v_dist_neighs.take(10)
