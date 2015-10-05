import pyspark
import itertools as it
sc = pyspark.SparkContext()

def copart(rdd1, rdd2):
    if rdd1.partitioner == rdd2.partitioner:
        return True
    else:
        return False
    return [(s, w) for s in neighbor_list]

def processNode(node):
    node1, node2 = node.split(': ')
    for i in node2.split(' '):
        node2 = int(to)
    return (int(node1), node2)    

def bfs(node_neighbor, node_name, iter = 10):
    # similiar to P4 bfs
    p_rdd = node_neighbor.filter(lambda (s,v): s == Kevin_Bacon)
    # Check we have the good one
    assert p_rdd.collect()[0][0] == Kevin_Bacon
    # check that p_rdd 
    assert copart(node_neighbor, p_rdd)

    for i in xrange(iter): 
        # check the copartition first
        assert copart(node_neighbor, p_rdd)
        # Create an rdd
        dir_node_neighbor = p_rdd.flatMapValues(lambda x: x).map(lambda g: (g[1], 1))
        # Use the same number of partitions as
        # node_neighbor, so they end up copartitioned.
        dir_node_neighbor = dir_node_neighbor.reduceByKey(lambda x, y: x + y, numPartitions 
                                                        = node_neighbor.getNumPartitions())
        # check that they are now copartitioned
        assert (dir_node_neighbor, node_neighbor)

        # Here we check if we have found Harvard_University
        if dir_node_neighbor.filter(lambda (s,v): s == Harvard_University).count() > 0:
            with open("P5.txt", "w") as f:
                f.write("Final Distance is ".format(i + 1) + '\n')
                f.write("\n\n")
            break

        p_rdd = node_neighbor.filter(lambda (s,v): s in dir_node_neighbor.keys().collect())


        print("Now the iteration is ".format(i))
        print("Distance is ".format(i + 1))

    return i + 1

if __name__ == '__main__':

    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
    node_name = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')    
    # process links
    node_neighbor = links.map(processNode)
    
    # create an RDD 
    nodes = zipWithIndex()
    node_name = node_name.nodes.map(lambda (i, j): (j + 1, i))
    node_name = node_name.sortByKey().cache()
    node_neighbor = node_neighbor.partitionBy(256).cache()
    
    #############################
    #########Kevin Bacon#########
    #############################
    Kevin_Bacon = node_name.filter(lambda (k, v): v == 'Kevin_Bacon').collect()
    assert len(Kevin_Bacon) == 1
    Kevin_Bacon = Kevin_Bacon[0][0]  # extract node id 
    #############################
    #######Harvard University####
    #############################
    Harvard_University = node_name.filter(lambda (K, V): V == 'Harvard_University').collect()
    assert len(Harvard_University) == 1
    Harvard_University = Harvard_University[0][0]  # extract node id

    # call bfs algorithm
    dis = bfs(node_neighbor, node_name, iter = 10)
    
    with open("P5.txt", "w") as f:
        f.write("Final Distance is " + str(dis) + '\n')
        f.write("\n\n")