import time
import findspark
findspark.init('/home/zelong/spark')
import pyspark
from P4_bfs import bfs
starttime = time.time()

sc = pyspark.SparkContext('local','bfs')
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


#generate txt file
f = open('P4.txt','w')

#write result of "CAPTAIN AMERICA"
output = bfs(graph_dist, 'CAPTAIN AMERICA', 6, sc )
result = output[0].cache()
num_iter = output[1] - 1
num_of_node = output[2]
f.write('Result of CAPTAIN AMERICA\n')
for i in range(num_iter):
	sentence = "Num of nodes has distance %d from CAPTAIN AMERICA: %d\n" % ( (i) , result.filter(lambda x: x[1][0] == (i+1)).count())
	f.write(sentence)

summary = "CAPTAIN AMERICA touched %d nodes in total, using %d iterations\n\n\n" % (num_of_node, num_iter)
f.write(summary)


#write result of "MISS THING/MARY"
output = bfs(graph_dist, 'MISS THING/MARY', 6, sc )
result = output[0].cache()
num_iter = output[1] - 1
num_of_node = output[2]
f.write('Result of MISS THING/MARY\n')
for i in range(num_iter):
	sentence = "Num of nodes has distance %d from MISS THING/MARY: %d\n" % ((i), result.filter(lambda x: x[1][0] == (i+1)).count())
	f.write(sentence)

summary = "MISS THING/MARY touched %d nodes in total, using %d iterations\n\n\n" % (num_of_node, num_iter)
f.write(summary)

#write result of "ORWELL"
output = bfs(graph_dist, 'ORWELL', 6, sc )
result = output[0].cache()
num_iter = output[1] - 1
num_of_node = output[2]
f.write('Result of ORWELL\n')
for i in range(num_iter):
	sentence = "Num of nodes has distance %d from ORWELL: %d\n" % ((i),  result.filter(lambda x: x[1][0] == (i + 1)).count())
	f.write(sentence)

summary = "ORWELL touched %d nodes in total, using %d iterations\n\n\n" % (num_of_node, num_iter)
f.write(summary)





