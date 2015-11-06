
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

#The following function is assumed that diameter is 10
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

#The following function is to use accumulator
#The benefit of accumulator is to use in parrallel threads
#The accumulator is to track the untouched nodes number
#everytime if one node is updated, the accumulator will be added one
#before every iteration,the accumulator will be set to 0
#in the end of iteration, we will check if accumulator is still 0
#if it is, ok, break the loop
"""
def bfs(graph,Root_name,sc):
    number_of_nodes = 0;
    graph = graph.map(lambda x: init_nodes(x,Root_name));
    accum = sc.accumulator(0)
    accum = 1;#set the accum to 1
    level = 0;
    #move to the next level
    #for level in range(10):
    while(accum.value):
        accum = 0;#set the accumulator to zero before every iteration
        #used rdd.filter to optimize the amount of data transffered
        #graph = graph.filter(lambda x: (x[1][0] == level - 1)).flatMap(lambda x: new_entries(x,level));
        graph = graph.flatMap(lambda x: new_entries(x,level));
        graph = graph.reduceByKey(lambda x,y: reconstruct(x,y));
        #check if the accum has changed
        if(accum.value == 0):
            break;
        level = level + 1;
    return graph.filter(lambda x: not(math.isinf(x[1][0]))).count()
"""