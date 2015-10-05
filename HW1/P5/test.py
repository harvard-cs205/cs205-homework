import time
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from P5_BFS import *
from  pyspark import SparkContext
sc = SparkContext()
sc.setLogLevel("ERROR")

def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [to.encode('utf-8') for to in dests.split(' ')]
    return (src.encode('utf-8'), dests)
    
text = sc.textFile("test.txt") 
neighbor_graph = text.map(link_string_to_KV)
edge_list = neighbor_graph.flatMap(lambda (K,V): [(K,V[i]) for i in range(len(V))])

print bfs_look2(edge_list,'1','10',20,sc)
