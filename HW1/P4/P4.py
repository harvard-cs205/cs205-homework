
# coding: utf-8

# In[1]:

import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="Spark1")
import re
#from P4_bfs import bfs
def bfs(sc, adj_matrix_rdd, char, diameter=float("inf")):
    '''
    Takes SparkContext (sc), adjacency matrix rdd (adj_matrix_rdd), character name (char), and optional diameter
    of graph as parameter and performs BFS.
    Note: adj_matrix_rdd contain 
        key = character name
        value = list of names of adjacent characters                   
    '''
    def group_neighbor_dist(neighbors,dist):
        return [(n,dist) for n in neighbors]
    
    dist = 0
    dists_rdd = sc.parallelize([(char,dist)]).partitionBy(32) #shortest paths to nodes
    neighbors_rdd = dists_rdd
    num_touched = 1
    #We assume graph diameter is <=10
    i = 0
    num_new_neighbors = 1 
    while (i < diameter) and (num_new_neighbors != 0):
        #get neighbors of neighbors
        neighbors_rdd = neighbors_rdd.join(adj_matrix_rdd) 
        #we only care about the new neighbors
        neighbors_rdd = neighbors_rdd.values()            .flatMap(lambda (prev_dist,neighbors): group_neighbor_dist(neighbors,prev_dist+1))            .distinct()            .partitionBy(32)
            
        #we now have rdd of (char,dist)
        assert neighbors_rdd.partitioner == dists_rdd.partitioner, "neighbors and dists are not copartitioned"
        #remove characters that we already have a shorter path to
        neighbors_rdd = neighbors_rdd.subtractByKey(dists_rdd) 
        neighbors_rdd.cache()
        num_new_neighbors = neighbors_rdd.count()
        num_touched += num_new_neighbors
        #update dists_rdd to include the new nodes we have explored
        dists_rdd = dists_rdd.union(neighbors_rdd)
        i += 1
        
    print num_touched
    return dists_rdd


# In[2]:

#source_rdd will contain (key=character, value=a comic that the character is in)
source_rdd = sc.textFile("source.csv",100)
cleaner_regex = re.compile('"(.+)","(.+)"')
source_rdd = source_rdd.map(lambda line: cleaner_regex.search(line).groups())


# In[3]:

#comic_rdd will contain (key=comic, value=list of characters in the comic)
comic_rdd = source_rdd.map(lambda (character, comic): (comic, [character]))
comic_rdd = comic_rdd.reduceByKey(lambda chars1, chars2: chars1 + chars2)


# In[4]:

#char_rdd will contain (key=character, value=set of comics the character is in)
char_rdd = source_rdd.map(lambda (character, comic): (character, set([comic])))
char_rdd = char_rdd.reduceByKey(lambda chars1, chars2: chars1.union(chars2))


# In[5]:

#now we want to make adj_matrix_rdd which contains 
#(key=character1, value=set of characters that appear in some comic with character1)

#dictionary where key=comic, value=list of characters in the comic
comic_dict = comic_rdd.collectAsMap()

def flatten (lst_of_lsts):
    #helper function that takes a list of list and returns a flattened list
    flat = []
    for l in lst_of_lsts:
        flat.extend(l)
    return flat

adj_matrix_rdd = char_rdd.map(lambda (char,comics): (char, list(set(flatten([comic_dict[comic] for comic in comics])))))
adj_matrix_rdd = adj_matrix_rdd.partitionBy(100)
adj_matrix_rdd.cache()


# In[ ]:

#BFS using RDDs
sources = ['CAPTAIN AMERICA','MISS THING/MARY','ORWELL']
#call into bfs
for source in sources:
    bfs(sc, adj_matrix_rdd, source, 10)


# In[ ]:



