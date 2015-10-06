import re
import time

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

root = "CAPTAIN AMERICA"

finished = sc.parallelize([(root, 0)])

last = time.time() 

for i in range(10): 
	print "\n\n\n"
	print "Iteration number %d" % i 
	print "Time: %d sec" % (time.time() - last)
	print "\n\n\n"
	last = time.time()

	finished_neighbors = finished.join(edge_lookup).flatMap(lambda (K,V): V[1])

	finished_neighbors = finished_neighbors.distinct().map(lambda x: (x, i+1), preservesPartitioning=True)
	possible_neighbors = finished_neighbors.leftOuterJoin(finished)

	new_elements = possible_neighbors.filter(lambda (K,V): V[1] == None).map(
		lambda (K,V): (K, V[0]), preservesPartitioning=True)


	finished = finished.union(new_elements).cache()

	print finished.count()

print "Finished Set: "
print finished.count()
print "\n\n"

