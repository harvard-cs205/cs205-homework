import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
sc = SparkContext()

###################
## SHORTEST PATH ##
###################

# Make page_names
# titles-sorted.txt
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt').zipWithIndex().map(lambda x: (x[1] + 1, x[0])).partitionBy(32).cache()

# Make page_links and flatten
# links-simple-sorted.txt
# from1: to11 to12 to13
page_links_with_strings = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt').map(lambda line: (int(line[:line.rindex(':')]), line[line.rindex(':') + 2:]))
# (1, 1664968), (2, 3), (2, 747213), (2, 1664968), (2, 1691047)
page_links = page_links_with_strings.map(lambda line: (line[0], [int(x) for x in line[1].split(' ')])).partitionBy(32).cache()

# Run some SSBFS searches
kevinToHarvard = path_finder('Kevin_Bacon','Harvard_University', page_links, page_names, sc)
harvardToKevin = path_finder('Harvard_University','Kevin_Bacon', page_links, page_names, sc)

# Just for fun
margaretToLizardPeople = path_finder('Margaret_Thatcher','Lizardfolk', page_links, page_names, sc)