import re
from P4_bfs import bfs

split_quote = re.compile(r'","')

data = sc.textFile('source.csv')

def clean_text(text): 
	stripped_first_last = text[1:len(text) - 1]

	# make sure that our strings are at least 2 chars 
	assert stripped_first_last != ''

	return tuple(split_quote.split(stripped_first_last)) 

char_com = data.map(clean_text)
com_char = char_com.map(lambda (x,y): (y,x)) 
com_all_chars = com_char.groupByKey()

char_chars = com_char.join(com_all_chars).values().map(lambda (k,v): (k, tuple(v)))
edge_lookup = char_chars.reduceByKey(lambda x,y : set(x) | set(y)).cache()

root = 'CARTER, PEGGY'

finished = sc.parallelize([(root, (0, [[root]]))])
new_size = sc.accumulator(0)

while True:
	i = 0 
	new_size.value = 0 
	print "Finished Size: ", finished.count()
	finished_neighbors = finished.filter(lambda (K,V): V[0] == i).join(edge_lookup).map(lambda (K,V): (K, (V[0][1], tuple(V[1]))))
	# mapping to get (neighbor, source), (neighbor, source) pairs 
	finished_neighbors = finished_neighbors.flatMap(lambda (K,V): zip(list(V[1]), [(K, V[0]) for _ in range(len(V[1]))]))
	finished_neighbors = finished_neighbors.map(lambda (dest,source): (dest, (source, i+1)), preservesPartitioning=True)

	print "Finished Neighbors, ", finished_neighbors.count()
	possible_neighbors = finished_neighbors.leftOuterJoin(finished)
	print "Possible Neighbors", possible_neighbors.count()
	def addNew(key, value):
		path = value[0][0][1]
		new_paths = [p + [key] for p in path]
		return (key, (value[0][1], new_paths))

	new_elements = possible_neighbors.filter(lambda (K,V): V[1] == None).map(lambda x: addNew(*x))

	print "New size", new_size.value
	new_elements.foreach(lambda x: new_size.add(1))
	print "New size", new_size.value
	finished = finished.union(new_elements).cache()

	if new_size.value == 0: 
		break 
	print finished.count()
	i += 1

finished.count()
