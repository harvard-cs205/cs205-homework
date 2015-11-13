import findspark
findspark.init()
import pyspark
import csv
import numpy as np 

def bfs(graph, source, sc):
    # converts the input adjacency_list representation to one with distances 
    # and flag 
    def map_adj_list(x):
        if x[0] == source:
            # want to explore from here the first round 
            # init distance to 0
            return (x[0], (1, 0, list(set(x[1])))) 
        else:
            return (x[0], (0, np.inf, list(set(x[1]))))

    # maybe use the preservesPartitioning flag here 
    d_adj_list = graph.map(map_adj_list)

    def closure():
        acc = sc.accumulator(0)
        # this will return the new adj_list with extra elements
        def mapper(x):
            node, v = x
            explored_flag, d, n_list = v
            if explored_flag == 1:
                # we need to explore this nodes neighbors 
                acc.add(1)
                # mark the current node as having its neighbors visited  
                v_list = [(node, (2, d, n_list))]
                for n in n_list:
                    # add in neighbors with flag == 1 so we explore them next round
                    v_list.append((n, (1, d+1, [])))
                return v_list
            else:
                # want to leave the node unchanged
                return [(node, v)]

        def reducer(v1, v2):
            explored_flag1, d1, n_lst1 = v1
            explored_flag2, d2, n_lst2 = v2
            # we take the max explored flag
            # min distance
            # concatenation of unique neighbors
            return max(explored_flag1, explored_flag2), min(d1, d2), list(set(n_lst1 + n_lst2))

        return acc, mapper, reducer 

    acc = sc.accumulator(1)
    while(acc.value > 0):
        acc, mapper, reducer = closure()
        # we need to do a flatMap for each node 
        # same key 
        d_adj_list = d_adj_list.flatMap(mapper)
        # force evaluation so we increment the accumulator 
        # then expand the nodes that should be explored right now
        d_adj_list = d_adj_list.reduceByKey(reducer)  
        d_adj_list.count()
    # could also filter for flag == 2 nodes 
    num_touched = d_adj_list.filter(lambda (n, (f, d, n_lst)): d < np.inf).count()
    return num_touched

