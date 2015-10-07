
# coding: utf-8

# In[ ]:

#Part 2 -- SSBFS
import math;
import findspark as fs;
fs.init();
import pyspark as py

def init_nodes(x,start_node):
    if(x[0] == start_node):
        return (x[0],(0,x[1]));
    return (x[0],(float('inf'),x[1]));

def new_entries(x,level):
    result = [];
    result.append(x);#still inf
    #find the parent level
    if (x[1][0] == level - 1):
        for entry in x[1][1]:
            result.append((entry,(level,[])));
    return result;
        
def reconstruct(x,y):
    level = min(x[0], y[0]);
    new_neighbor = set(x[1] + y[1]);
    return (level,list(new_neighbor));

def bfs(graph,Root_name,sc):
    accum = sc.accumulator(0);
    number_of_nodes = 0;
    graph = graph.map(lambda x: init_nodes(x,Root_name));
    #move to the next level
    for level in range(10):
        accum += 1;
        #used rdd.filter to optimize the amount of data transffered
        #graph = graph.filter(lambda x: (x[1][0] == level - 1)).flatMap(lambda x: new_entries(x,level));
        graph = graph.flatMap(lambda x: new_entries(x,level));
        graph = graph.reduceByKey(lambda x,y: reconstruct(x,y));
    return graph.filter(lambda x: not(math.isinf(x[1][0]))).count()

