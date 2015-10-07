import findspark
findspark.init()
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName('P4').setMaster('local')
sc = SparkContext(conf=conf)

# Graph and BFS to determine the number of touched nodes of a signle-sorce search

def helper(x, y):
	if y[0] != y[1]:
		return (x, y)

def getGraph(source):
	source_p = sc.textFile(source)
	comic_char = source_p.map(lambda line: (line.split('"' or ',')[3], line.split('"' or ',')[1]))
	comic_two_char = comic_char.join(comic_char)
	comic_two_char_distinct = comic_two_char.filter(lambda (x, y): helper(x, y))
	two_char_distinct = comic_two_char_distinct.map(lambda (x, y): y)
	return two_char_distinct

graph = getGraph('./source.csv')

def BFS(start):
	RDD_iter = sc.parallelize([(start, 0)])
	RDD_visited = RDD_iter
	while True:
		RDD_iter = graph.join(RDD_iter).distinct().map(lambda (x, y): (y[0], 0)).distinct().subtract(RDD_visited)
		RDD_visited = RDD_visited.union(RDD_iter)
		if RDD_iter.count() == 0: break
	# Use of accumulators doesn't make much sense here; return RDD_visited.count() accomplishes the same
	accum = sc.accumulator(0)
	RDD_visited.foreach(lambda x: accum.add(1))
	return accum.value

nodes_num = BFS('ORWELL')
print nodes_num