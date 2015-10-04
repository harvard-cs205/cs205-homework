import pyspark
from P5_bfs import process_links, get_by_key, bfs_path

# initialize spark context
conf = pyspark.SparkConf().setAppName("P5")
sc = pyspark.SparkContext(conf=conf)

# get link and title RDDS from text files
links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')

link_graph = links.map(process_links).partitionBy(8).cache()
names_to_ids = page_names.zipWithIndex().mapValues(lambda v: int(v) + 1).cache()
ids_to_names = names_to_ids.map(lambda (k,v): (v,k)).partitionBy(8).cache()

# find paths linking Kevin Bacon & Harvard University in both directions
bacon = get_by_key('Kevin_Bacon', names_to_ids)
harvard = get_by_key('Harvard_University', names_to_ids)

bacon_to_harvard = [get_by_key(idx, ids_to_names) for idx in bfs_path(sc, link_graph, bacon, harvard)]
harvard_to_bacon = [get_by_key(idx, ids_to_names) for idx in bfs_path(sc, link_graph, harvard, bacon)]

print bacon_to_harvard
print harvard_to_bacon
