import pyspark 
import numpy as np 
import csv 
import time 

def bfs(graph, source, target, sc):
    def map_adj_list(x):
         # converts the input adjacency_list representation to one with distances 
         # and flag 
        if x[0] == source:
            # want to explore from here the first round 
            # init distance to 0
            # init previous node to None 
            return (x[0], (1, 0, None, list(set(x[1])))) 
        else:
            return (x[0], (0, np.inf, None, list(set(x[1]))))

    def closure():
        seen_target = sc.accumulator(0)
        # this will return the new adj_list with extra elements
        # when we add the target we should hit an accumulator 
        def mapper(x):
            node, v = x
            explored_flag, d, prev, n_list = v
            if explored_flag == 1:
                # we need to explore this nodes neighbors 
                #acc.add(1)
                # mark the current node as having its neighbors visited  
                v_list = [(node, (2, d, prev, n_list))]
                for n in n_list:
                    # add in neighbors with flag == 1 so we explore them next round
                    if n == target: 
                        seen_target.add(1)
                    v_list.append((n, (1, d+1, node, [])))
                return v_list
            else:
                # want to leave the node unchanged
                return [(node, v)]
        def reducer(v1, v2):
            explored_flag1, d1, prev1, n_lst1 = v1
            explored_flag2, d2, prev2, n_lst2 = v2
            # need to see which prev to keep 
            if d1 < d2: 
                prev = prev1
            else:
                prev = prev2 
            return max(explored_flag1, explored_flag2), min(d1, d2), \
                        prev, list(set(n_lst1 + n_lst2))
        return seen_target, mapper, reducer 

    def construct_path(prev_nodes, node):
        # prev_nodes is a dict keyed on nodes with value being the previous node
        prev = prev_nodes[node]
        # stop backtracing when we hit the source 
        if prev == source:
            return [node, prev]
        else:
            return [node] + construct_path(prev_nodes, prev)
    
    # maybe use the preservesPartitioning flag here 
    d_adj_list = graph.map(map_adj_list, preservesPartitioning=True).cache()
    # will store the number of new nodes
    #acc = sc.accumulator(1)
    seen_target = sc.accumulator(0)
    while(seen_target.value == 0):
        # add acc back
        seen_target, mapper, reducer = closure()
        # we need to do a flatMap for each node 
        d_adj_list = d_adj_list.flatMap(mapper, preservesPartitioning=True)
        # force evaluation so we increment the accumulator 
        # then expand the nodes that should be explored right now
        d_adj_list = d_adj_list.reduceByKey(reducer, numPartitions=64)  
        # force the computation for this level
        d_adj_list.count()

    reachable_graph = d_adj_list.filter(lambda (n, (f, d, prev, n_lst)): d < np.inf).cache()
    prev_nodes = dict(reachable_graph.map(lambda (n, (f, d, prev, n_lst)): (n, prev), preservesPartitioning=True).collect())
    path = construct_path(prev_nodes, target)
    return path

def link_mapper(x):
    # convert links to integers
    link, o_links = x.split(":")  
    return (int(link), map(lambda x: int(x), o_links.split(" ")[1:]))

def main():
    sc = pyspark.SparkContext()
    sc.setLogLevel("WARN")

    # links are strings 
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
    # we could make name_inds strings if we wanted but we 
    name_inds = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt').zipWithIndex().map(lambda (k, v): (k, v+1))
    r_name_inds = name_inds.map(lambda (k, v): (v, k)).cache()

    hu_id = name_inds.lookup("Harvard_University")[0]
    bac_id = name_inds.lookup("Kevin_Bacon")[0]

    adj_list = links.map(link_mapper).cache()
    
    id_path = bfs(adj_list, bac_id, hu_id, sc)
    # iterate through path and find the names 
    path = []
    for node in id_path[::-1]: 
        # lookup them up in the reverse name_inds table 
        path.append(r_name_inds.lookup(node)[0])

    print "Bacon to Harvard"
    print path


if __name__=="__main__":
    main()
