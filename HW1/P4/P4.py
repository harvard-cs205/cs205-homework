import findspark
findspark.init()
import pyspark
import json
from P4_bfs import bfs
import time
sc = pyspark.SparkContext(appName="P4")
start = time.time()
c = sc.textFile('source.csv')

# This function creates a tuple
# input----word: the word in the .txt file
# ouput----a tuple in the required format
def preprocess_mapper(x):
    x = x.split('","')   
    if len(x) == 2:
        character = x[0][1:]
        comic = x[1][:-1]
        return (comic, character)
    return (-1, -1)

movie_characters_RDD = c.map(preprocess_mapper).filter(lambda (k, v): v != -1).groupByKey().mapValues(lambda x: list(x))

# This function finds the neighbors of a node
# input----K,V pair: a key,value pair. 
# ouput----a list of neighbors
def neighbors_mapper((key, value)):
    res = []
    for i in range(0, len(value)):
        res.append((value[i], value[:i] + value[i+1:]))
    return res
    
character_neighbors_RDD = movie_characters_RDD.flatMap(neighbors_mapper)

# This function finds the neighbors of a node
# input----K,V pair: a key,value pair. 
# ouput----a list of neighbors
def combine_neighbors_reducer(neig1, neig2):
    for i in range(0, len(neig2)):
        if neig2[i] not in neig1:
            neig1.append(neig2[i])
    return neig1

pregraphRDD = character_neighbors_RDD.reduceByKey(combine_neighbors_reducer)

# This function finds the neighbors of a node
# input----K,V pair: a key,value pair. 
# ouput----a FILO queue
def graph_mapper((key, value)):
    value.insert(0, key)
    return value

graphRDD = pregraphRDD.map(graph_mapper)
graphRDD = graphRDD.filter(lambda x:len(x)>1)
character = 'CAPTAIN AMERICA','MISS THING/MARY','ORWELL'
log_file = open('P4_log.txt', 'w')
res = bfs(sc,graphRDD, character[0]),bfs(sc,graphRDD, character[1]),bfs(sc,graphRDD, character[2])
log_file.write('Origin = '+character[0]+'\nThe diameter is'+str(res[0][0])+'\nThe total number of nodes touched:'+str(res[0][1])+'\n'+res[0][2]+'\n\n\n')
log_file.write('Origin = '+character[1]+'\nThe diameter is'+str(res[1][0])+'\nThe total number of nodes touched:'+str(res[1][1])+'\n'+res[1][2]+'\n\n\n')
log_file.write('Origin = '+character[2]+'\nThe diameter is'+str(res[2][0])+'\nThe total number of nodes touched:'+str(res[2][1])+'\n'+res[2][2]+'\n\n\n')


log_file.close()
print '\nthe running time is', time.time()-start
