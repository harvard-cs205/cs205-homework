from pyspark import SparkContext, SparkConf
sc = SparkContext()

num_partitions = 256

# Make page_links and flatten
# from1: to11 to12 to13
page_links_with_strings = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt').map(lambda line: (int(line[:line.rindex(':')]), line[line.rindex(':') + 2:]))

# (1, [1664968]), (2, [3, 4])
page_links = page_links_with_strings.map(lambda line: (line[0], [int(x) for x in line[1].split(' ')])).partitionBy(num_partitions).cache()
print "Total number of original nodes: ", page_links.keys().count()


def lesserFirst(x):
    tmp_list = combineLists([x[0]], x[1])
    return (min(tmp_list), tmp_list)

def combineLists(x,y):
    new_list = list(set(x)|set(y))
    return new_list
     
def combinePotentialLists(x,y):
    new_list = []
    if len(x) > 0:
        new_list = x
    if len(y) > 0:
        for yi in y:
            if yi not in new_list:
                new_list.append(yi)
    return new_list

def connectedComponents(symmetric_connected_node_list, sc):
    nodes = symmetric_connected_node_list.partitionBy(num_partitions).cache()
    nodes = nodes.map(lambda x: (x[0], min(x[1])))    

    num_components = nodes.count()
    
    while True:
        print "Starting a new iteration with %s components" % num_components
        nodes_r = nodes.map(lambda x: (x[1], x[0]))
        nodes_union = nodes.union(nodes_r).groupByKey().map(lambda x: lesserFirst(x))
        nodes_reflattened = nodes_union.flatMapValues(lambda x: x).map(lambda x: (x[1], x[0])).reduceByKey(lambda x, y: min(x, y))
        nodes = nodes_reflattened.partitionBy(num_partitions).cache()
        new_num_components = nodes.values().distinct().count()
        if new_num_components == num_components:
            print "Didn't reduce the number of components this turn, breaking!"
            break
        num_components = new_num_components
        
    # Sum up all the number of values that appear, essentially how many nodes are in each component
    node_values = nodes.values().map(lambda x: (x, 1)).reduceByKey(lambda x,y: x+y).takeOrdered(1, key=lambda x: -x[1])
    print "Finished Collecting Components!"
    print "FINAL NUMBER OF COMPONENTS IS: %s " % num_components 
    print "COMPONENT WITH MOST NODES HAS %s NODES (LOWEST LINK ID IS %s)" % (node_values[0][1], node_values[0][0])
    print "TOTAL NUM NODES TOUCHED %s " % nodes.count()
    return nodes
    
# Counts as an edge even if the link goes in only one direction
def uniConnectedComponents(page_links, sc):
    print "Running Uni-Directional Connected Components"
    symmetric_connected_node_list = page_links.map(lambda x: (x[0], (x[1] + [x[0]]))).union(page_links.flatMapValues(lambda x: x).map(lambda x: (x[1], [x[0], x[1]])))    
    
    return connectedComponents(symmetric_connected_node_list, sc)
    
# Only counts as an edge if the link goes in both directions
def biConnectedComponents(page_links, sc):
    print "Running Bi-Directional Connected Components"
    page_links = page_links.map(lambda x: (x[0], (x[1] + [x[0]])))
    nodes = page_links.join(page_links.flatMapValues(lambda x: x)).map(lambda x: (x[0], combinePotentialLists(x[1][0], [x[1][1]])))    
    
    nodes = connectedComponents(nodes, sc)
    return nodes


#########################
## LINKS ARE SYMMETRIC ##
#########################

# Run uni-directional Connected Components, where a link only needs to exist in one direction to count
print "##########################################"
uni_nodes = uniConnectedComponents(page_links, sc)
print "##########################################"


#############################
## LINKS ARE NOT SYMMETRIC ##
#############################

# Run bi-directional Connected Components, where a link must exist in both directions to count
print "##########################################"
bi_nodes = biConnectedComponents(page_links, sc)
print "##########################################"