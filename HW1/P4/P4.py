import findspark
findspark.init('/Users/george/Documents/spark-1.5.0')
from pyspark import SparkContext
from P4_bfs import *

sc = SparkContext()
source = sc.textFile("source.csv")

# return comic -> char
def clean_csv(x):
    c = x[1:-1].split('","')
    return (c[1], c[0])

# return comic -> [char]
def clean_csv_list(x):
    c = x[1:-1].split('","')
    return (c[1], [c[0]])

# create mapping of char -> adj list
def create_neighbors(ele):
    char = ele[1]
    neighbors = char[1][:]

    # remove self edge
    neighbors.remove(char[0])
    return (char[0], set(neighbors))

# create nodes
nodes = source.map(clean_csv)
nodes_list = source.map(clean_csv_list) \
                   .reduceByKey(lambda x,y: x + y)

# create graph
graph = nodes.join(nodes_list).map(create_neighbors)
graph = graph.partitionBy(16)

# print out answer
characters = ['CAPTAIN AMERICA', 'MISS THING/MARY', 'ORWELL']
for char in characters:
    print char + " " + str(ss_bfs(sc, graph, char))

