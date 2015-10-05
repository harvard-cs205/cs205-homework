import pyspark
sc = pyspark.SparkContext()
from P4_bfs import p4bfs
import itertools as it

characters = [u'CAPTAIN AMERICA', u'ORWELL', u'MISS THING/MARY']
# Load data
source = sc.textFile('source.txt')

# Convert into (keys, values)
source = source.map(lambda x : x.split('"'))
source_list = source.map(lambda x : (x[3], x[1])).groupByKey()

# Create a rdd with keys equals to characters and values equals to neighbords
new_source = source_list.values().map(lambda x: [i for i in x])
new_source = new_source.map(lambda x: list(it.permutations(x,2))).map(lambda x: 
	(1, list([i for i in x]))).flatMapValues(lambda x: x).values().groupByKey().sortByKey()
nodes = new_source.map(lambda v: (v[0], [i for i in v[1]]))

# CAPTAIN AMERICA
node1, node2, distance1 = p4bfs(nodes, characters[0])
# ORWELL
node3, node4, distance2 = p4bfs(nodes, characters[1])
# MISS THING/MARY
node5, node6, distance3 = p4bfs(nodes, characters[2])

with open("P4.txt", "w") as f:
	f.write("For BFS -- CAPTAIN AMERICA" + '\n')
	f.write("Distance is " + str(distance1) + '\n')
	f.write("Number of connected characters is " + str(node1 - node2))
	f.write("\n\n")
	f.write("For BFS -- ORWELL" + '\n')
	f.write("Distance is " + str(distance2) + '\n')
	f.write("Number of connected characters is " + str(node3 - node4))
	f.write("\n\n")
	f.write("For BFS -- MISS THING/MARY" + '\n')
	f.write("Distance is " + str(distance3) + '\n')
	f.write("Number of connected characters is " + str(node5 - node6))