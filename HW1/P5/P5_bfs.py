import pyspark
import pdb
import itertools as it

# shut down the previous spark context
sc = pyspark.SparkContext()
sc.setLogLevel('WARN')





def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)


def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner
    return [(n, w) for n in neighbor_list]

def find_distance(neighbors,
             page_names,
             iterations=10):
    ''' run the BFS algorithm'''

    # like in P4_bfs, set 1 to the values of the nodes catalog
    pages_rdd = neighbors.filter(lambda (k,v): k == Kevin_Bacon)
    #pages_rdd = neighbors.mapValues(lambda _: 1.0)

    # Check we have the good one
    assert pages_rdd.collect()[0][0] == Kevin_Bacon
    # check that pages_rdd 
    assert copartitioned(neighbors, pages_rdd)



    for i in range(iterations): # not while loop to make sure the code stops

        # First make sure they are copartitioned.
        assert copartitioned(neighbors, pages_rdd)


        # Create an rdd with [ (neighbor1,1), (neighbor2,1)  ]
        # However direct_neighbors, neighbors and pages_rdd are no longer copartitioned
        direct_neighbors = pages_rdd.flatMapValues(lambda x: x).map(lambda g: (g[1],1))

        # Use the same number of partitions as
        # neighbors, so they end up copartitioned.
        direct_neighbors = direct_neighbors.reduceByKey(lambda x, y: x + y, numPartitions=neighbors.getNumPartitions())

        # check that they are now copartitioned
        assert (direct_neighbors,neighbors)

        # Here we check if we have found Harvard_University
        if direct_neighbors.filter( lambda (k,v) : k == Harvard_University  ).count() > 0:
            print 'the search is over !'
            print("Final Distance {0}".format(i+1))
            break

        # Otherwise we will look at the neighbors of these neighbors in the next
        # iteration
        list_neighbors = direct_neighbors.keys().collect()
        pages_rdd = neighbors.filter( lambda (k,v): k in list_neighbors )


        # Report current iteration == distance
        print("Iteration {0}".format(i))
        print("= Distance {0}".format(i+1))

    return i+1

if __name__ == '__main__':

	#links = sc.textFile('links-simple-sorted.txt',32)
	links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt',32)
	
	#page_names = sc.textFile('titles-sorted.txt', 32)
	page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt',32)
	
	
	# process links into (node #, [neighbor node #, neighbor node #, ...]
	neighbors = links.map(link_string_to_KV)
	
	# create an RDD for looking up page names from numbers
	# First start the index at 1 and rdd = (index, title) and sortbykey
	page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
	page_names = page_names.sortByKey().cache()
	
	#######################################################################
	# set up partitioning - we have roughly 16 workers, if we're on AWS with 4
	# nodes not counting the driver.  This is 16 partitions per worker.
	#
	# Cache this result, so we don't recompute the link_string_to_KV() each time.
	#######################################################################
	neighbors = neighbors.partitionBy(256).cache()
	
	
	# find Kevin Bacon
	Kevin_Bacon = page_names.filter(lambda (k, v): v == 'Kevin_Bacon').collect()
	# This should be [(node_id, 'Kevin_Bacon')]
	assert len(Kevin_Bacon) == 1
	Kevin_Bacon = Kevin_Bacon[0][0]  # extract node id
	
	# find Harvard University
	Harvard_University = page_names.filter(lambda (K, V):
                                       	V == 'Harvard_University').collect()
	# This should be [(node_id, 'Harvard_University')]
	assert len(Harvard_University) == 1
	Harvard_University = Harvard_University[0][0]  # extract node id


	# run the algorithm to find the distance
	distance = find_distance(neighbors,
        	page_names,
        	iterations=10) # this is the max diameter

	print distance





# commands to run on the cluster
#  aws emr create-cluster --name "Spark cluster" --release-label emr-4.1.0 --applications Name=Spark --ec2-attributes KeyName=mykeypair --instance-type m3.xlarge --instance-count 5 --use-default-roles

# spark-submit --num-executors 4 --executor-cores 4 --executor-memory 8g P5.py










