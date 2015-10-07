from pyspark import SparkContext
from P5_bfs import ssBFS2

# Setup files and context
sc = SparkContext("local", "P5")
links = sc.textFile ('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
page_names = sc.textFile ('s3://Harvard-CS205/wikipedia/titles-sorted.txt')

# Set up logger so as to be able to see output
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org").setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

# Functor to create a component for the graph by parsing provided link string
def createGraphComponent(link):
	frm, to = link.split(": ")
	return (int(frm), map(int, to.split(" ")))

# Use functor to create a graph of links similar to P4 graph
graph = links.map(lambda link: createGraphComponent(link)).cache()
print graph.collect()

# Zip up page names and take into account 1-indexing
page_names = page_names.zipWithIndex().map(lambda (page, i): (page, i+1)).sortByKey()
# Use this one for index lookups
page_names_idx = page_names.zipWithIndex().map(lambda (page, i): (i+1, page)).sortByKey()

# Our two start/end nodes - Kevin and Harvard
kevin = page_names.lookup("Kevin_Bacon")[0]
harvard = page_names.lookup("Harvard_University")[0]

# Run shiny new BFS2 to get shortest paths, K->H
# [[src, elt1, ...], [src, elt2, ...], ...]
paths1 = ssBFS2(graph, kevin, harvard, sc)

# Get named paths instead of indices
namedPaths1 = []
for path in paths1:
	namedPath = map(lambda elt: page_names_idx.lookup(elt), path)
	namedPath.append(u'Harvard_University') # Looks like the exit node is not returned
	namedPaths1.append(namedPath)
print namedPaths1

# Run shiny new BFS2 to get shortest paths, H->K
# [[src, elt1, ...], [src, elt2, ...], ...]
paths2 = ssBFS2(graph, harvard, kevin, sc)

# Get named paths instead of indices
namedPaths2 = []
for path in paths2:
	namedPath = map(lambda elt: page_names_idx.lookup(elt), path)
	namedPath.append(u'Kevin_Bacon') # Looks like the exit node is not returned
	namedPaths2.append(namedPath)
print namedPaths2