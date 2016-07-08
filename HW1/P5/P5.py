# from https://github.com/thouis/SparkPageRank/blob/master/PageRank.py modified
# helper function to get a graph rdd
def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

# process links into (node #, [neighbor node #, neighbor node #, ...]
neighbor_graph = links.map(link_string_to_KV)

# create an RDD for looking up page names from numbers
# remember that it's all 1-indexed
page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
page_names = page_names.sortByKey().cache()

#######################################################################
# set up partitioning - we have roughly 4 workers, if we're on AWS with 4
# nodes not counting the driver.  This is 8 partitions per worker.
# As we do an union afterwards we don't want it to be to big
# Cache this result, so we don't recompute the link_string_to_KV() each time.
#######################################################################

neighbor_graph = neighbor_graph.partitionBy(32).cache()

# find Kevin Bacon
Kevin_Bacon = page_names.filter(lambda (K, V): V == 'Kevin_Bacon').collect()
# This should be [(node_id, 'Kevin_Bacon')]
assert len(Kevin_Bacon) == 1
Kevin_Bacon = Kevin_Bacon[0][0]  # extract node id

# find Harvard University
Harvard_University = page_names.filter(lambda (K, V):
                                       V == 'Harvard_University').collect()
# This should be [(node_id, 'Harvard_University')]
assert len(Harvard_University) == 1
Harvard_University = Harvard_University[0][0]  # extract node id

# we are now done extracting the graph

print "distance from Kevin_Bacon to Harvard_University is ", distance_between(sc,neighbor_graph , Kevin_Bacon, Harvard_University)

print "distance from Harvard_University  to Kevin_Bacon  is ", distance_between(sc,neighbor_graph ,Harvard_University , Kevin_Bacon )
