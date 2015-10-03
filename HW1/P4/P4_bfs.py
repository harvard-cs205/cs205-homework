import pyspark
from pyspark import SparkContext, SparkConf
#sc = SparkContext()

def bfs(graph, starting_char):
    char_list = [starting_char]
    BFS_list  = []
    
    while len(char_list)!=0:
        neighbors = graph.filter(lambda tpl: tpl[0] in char_list).flatMap(lambda tpl: tpl[1]).collect()
        BFS_list  = BFS_list + char_list
        char_list = []
        char_list = char_list + list(set(neighbors)-set(BFS_list))                                     
        print len(list(set(neighbors)-set(BFS_list)))
    #print len(BFS_list)
    


