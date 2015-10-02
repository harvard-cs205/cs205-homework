import findspark
findspark.init()
import pyspark

# num partitions (4 times the number of cores)
N = 16

def bfs(root, graph):
	# initialize queue and distances rdds
	distances = queue = sc.parallelize([(root, 0)], N)
	acc = sc.accumulator(1)
	iteration = 0
	while acc.value != 0:
		acc = sc.accumulator(0)
		iteration += 1
		# keep a temporary copy of the queue to later on remove items in queue from graph
		temp = queue
		# get all (neighbor, distance) value pairs for each key in queue
		joined = graph.join(queue, N).distinct()
		# have to partition again when taking the values
		queue = joined.values().distinct().partitionBy(N)
		# take out original queue keys from graph and cache for later iterations
		graph = graph.subtractByKey(temp, N).cache()
		# set the distance values to current iteration value
		newDistances = queue.mapValues(lambda _: iteration)
		# remove already visited keys from the queue, and cache for later iterations
		queue = newDistances.subtractByKey(distances, N).cache()
		# add keys in queue with their distances to distances rdd, and cache for later iterations
		distances = distances.union(queue).partitionBy(N).cache()
		# increment accumulator to satisfy the while condition (unless the queue is empty, in which case the loop stops after this)
		queue.foreach(lambda _: acc.add(1))
	return (distances.count(), distances)

sc = pyspark.SparkContext()

# make pyspark shut up
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

source = sc.textFile("source.csv", N)
# take care of formatting, end up with tuple of (comic, character)
comic_char = source.map(lambda lst: tuple(lst.split('","')))\
	.map(lambda pair: (pair[1].replace('"', ''), pair[0].replace('"', '')))

# join comic_char with itself on comic keys, to get characters from the same comic associated with each other
joined = comic_char.join(comic_char, N)
# filter out duplicates and edges between a character and itself, and cache for later use
graph = joined.values().distinct().filter(lambda (u,v): u!= v).partitionBy(N).cache()

roots = ["CAPTAIN AMERICA", "MISS THING/MARY", "ORWELL"]
for root in roots:
	(num_touched, distances_rdd) = bfs(root, graph)
	print num_touched


