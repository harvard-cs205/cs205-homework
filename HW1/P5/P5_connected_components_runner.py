import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
sc = SparkContext()
import P5_connected_components

num_partitions = 8

# Make page_links and flatten
# from1: to11 to12 to13
# s3://Harvard-CS205/wikipedia/
page_links_with_strings = sc.textFile('links-simple-sorted.txt').map(lambda line: (int(line[:line.rindex(':')]), line[line.rindex(':') + 2:]))

# (1, [1664968]), (2, [3, 4])
page_links = page_links_with_strings.map(lambda line: (line[0], [int(x) for x in line[1].split(' ')])).partitionBy(num_partitions).cache()
print "Total number of original nodes: ", len(page_links.keys().collect())



#########################
## LINKS ARE SYMMETRIC ##
#########################

# Run uni-directional Connected Components, where a link only needs to exist in one direction to count
print "##########################################"
uni_nodes = P5_connected_components.uniConnectedComponents(page_links, sc, num_partitions)
print "##########################################"



#############################
## LINKS ARE NOT SYMMETRIC ##
#############################

# Run bi-directional Connected Components, where a link must exist in both directions to count
print "##########################################"
bi_nodes = P5_connected_components.biConnectedComponents(page_links, sc, num_partitions)
print "##########################################"