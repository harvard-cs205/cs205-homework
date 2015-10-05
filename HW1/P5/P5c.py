import findspark
findspark.init()
import pyspark
import csv
import numpy as np 

# we need to update the graph to store the previous neighbor 

sc = pyspark.SparkContext()
sc.setLogLevel('WARN')
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

def bfs(graph, source='ORWELL', target):
    # converts the input adjacency_list representation to one with distances 
    # and flag 
    def map_adj_list(x):
        if x[0] == source:
            # want to explore from here the first round 
            # init distance to 0
            # init previous node to None 
            return (x[0], (1, 0, None, list(set(x[1])))) 
        else:
            return (x[0], (0, np.inf, None, list(set(x[1]))))

    # maybe use the preservesPartitioning flag here 
    d_adj_list = graph.map(map_adj_list)

    def closure():
        acc = sc.accumulator(0)
        # this will return the new adj_list with extra elements
        # when we add the target we should hit an accumulator 
        def mapper(x):
            node, v = x
            explored_flag, d, prev, n_list = v
            if explored_flag == 1:
                # we need to explore this nodes neighbors 
                acc.add(1)
                # mark the current node as having its neighbors visited  
                v_list = [(node, (2, d, prev, n_list))]
                for n in n_list:
                    # add in neighbors with flag == 1 so we explore them next round
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

        return acc, mapper, reducer 
    
    def construct_path(prev_nodes, node):
        # prev_nodes is a dict keyed on nodes with value being the previous node
        prev = prev_nodes[node]
        # stop backtracing when we hit the source 
        if prev == source:
            return [prev]
        else:
            return [prev] + construct_path(prev_nodes, prev)

    acc = sc.accumulator(1)
    while(acc.value > 0):
        acc, mapper, reducer = closure()
        # we need to do a flatMap for each node 
        # same key 
        d_adj_list = d_adj_list.flatMap(mapper)
        # force evaluation so we increment the accumulator 
        d_adj_list.count()
        # then expand the nodes that should be explored right now
        d_adj_list = d_adj_list.reduceByKey(reducer)  
    # could also filter for flag == 2 nodes 
    num_touched = d_adj_list.filter(lambda (n, (f, d, prev, n_lst)): d < np.inf).count()
    # we also want to return the path 
    prev_nodes = dict(d_adj_list.map(lambda (n, (f, d, prev, n_lst)): (n, prev)).collect())
    path = construct_path(prev_nodes, target)
    return num_touched

# construct the graph
edges = issue_sh.flatMap(construct_edges).distinct().map(lambda x: (x[0][0], [x[0][1]]))
adj_list = edges.reduceByKey(lambda x, y: x + y)
print bfs(adj_list, "CAPTAIN AMERICA")
