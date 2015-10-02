

import time
import findspark
findspark.init('/home/zelong/spark')
import pyspark
import P4_bfs

#starttime = time.time()

sc = pyspark.SparkContext()
# Construct the graph from data
data = sc.textFile('source.csv')
pair = data.map(lambda x: x[1:-1].split("\",\""))
movie_name_pair = pair.map(lambda x: (x[1],x[0]))
group_by_movie = movie_name_pair.groupByKey().map(lambda x: (x[0],list(set(x[1]))))

def assign_neighbor(x):
    # x is an iterable contains all characters in the movie
    result = [] 
    for i in range(len(x)):
	key = x[i]
	value = x[0:i] + x[i+1:]
	result.append( (key, value) )
    return result 

neighbor = group_by_movie.flatMap(lambda x: assign_neighbor(x[1]))
graph = neighbor.groupByKey().map(lambda x: (x[0], list(set([i for j in x[1] for i in j]))))
graph_dist = graph.map(lambda x: (x[0], ( 999999 , x[1]   )))



output = P4_bfs.bfs(graph_dist, 'CAPTAIN AMERICA', 4 )
result = output[0]
num_iter = output[1]
#endtime = time.time()
print result.filter(lambda x: x[1][0] == 1).count()
print result.filter(lambda x: x[1][0] == 2).count()
print result.filter(lambda x: x[1][0] == 3).count()

#print "TIMe", endtime - starttime
#print num_iter


#print P4_bfs.bfs(graph_dist, 'CAPTAIN AMERICA', 10 ).lookup('CAPTAIN AMERICA')[0][0]
#print sorted(P4_bfs.bfs(graph_dist, 'CAPTAIN AMERICA', 10 ).lookup('VAPOR')[0][1]) == sorted(graph_dist.lookup('VAPOR')[0][1])
#print graph_dist.lookup('VAPOR')[0][0]
#print P4_bfs.bfs(graph_dist, 'CAPTAIN AMERICA', 10 ).take(2)

#print len(graph_dist.lookup('CAPTAIN AMERICA')[0][1])
#print graph_dist.keys().take(5)
#print group_by_movie.collect()[0]
#print assign_neighbor(group_by_movie.collect()[0][1])
#print neighbor.take(5)
#s = str(result_rdd.max(key = compare))
#print data.count()
#print pair.collect()[27]



