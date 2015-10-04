import pyspark
from operator import add, or_
from P4_bfs import process_line, ss_bfs

# initialize spark context
sc = pyspark.SparkContext("local[4]", "Spark1")

# get contents of text file as (comic, character) pairs
raw = sc.textFile('source.csv')
parsed = raw.map(process_line)

# convert to (comic, list of characters) pairs
lst_pairs = parsed.reduceByKey(add)

# get character connections
dup_chars = lst_pairs.flatMap(lambda (k,lst): [(char,set([c for c in lst if c != char])) for char in lst])
char_neighbors = dup_chars.reduceByKey(or_).partitionBy(8).cache()

# perform single-source breadth-first search for Cap, Miss Thing, & Orwell
cap = ss_bfs(sc, char_neighbors, 'CAPTAIN AMERICA')
miss_thing = ss_bfs(sc, char_neighbors, 'MISS THING/MARY')
orwell = ss_bfs(sc, char_neighbors, 'ORWELL')

print 'Cap: touched {} nodes'.format(cap.count())
print 'Miss Thing: touched {} nodes'.format(miss_thing.count())
print 'Orwell: touched {} nodes'.format(orwell.count())
