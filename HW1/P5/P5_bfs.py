# I throw in the towel.  This is a mess and I am not close to finishing.  

import findspark
findspark.init()
import pyspark

def BFS(graph, names, iterations=10, inspect=[], sc):
    
    # Rdd of path that will be added to.  
    rdd = sc.parallelize([(inspect[0],0)], 2).cache()
    
    # This RDD is used intermediately to build a collection of connections to build on
    namesRdd = rdd

    # Iterator used in the while loop
    i = 0
       
    acc = sc.accumulator(0)    

    while acc.value == 0 and i <= iterations:

        i+=1

        # Use join to look up the connections of the next round of connections
        namesRdd = graph.join(namesRdd)

        # Use mapValues to take just the list of connections and remove the distance
        namesRdd = namesRdd.map(lambda x: (x[0], x[1][0])).partitionBy(2)

        # Take the connections and flatMap them into one big pool to consider
        namesRdd = namesRdd.values().flatMap(lambda x: x)

        # Reduce the connections to distinct entries
        namesRdd = namesRdd.distinct()
        
        # Subtract out the names that are already in the rdd
        namesRdd = namesRdd.subtract(rdd.keys())
        
        if namesRdd.isEmpty(): acc = sc.accumulator(1)   # If nothing was added, change the accumulator value to exit the loop
        else:
            # create tuples of the connection name and the current distance of the iteration
            namesRdd = namesRdd.map(lambda x: (x, i)).partitionBy(2).cache()

            # Union the connection tuples to the existing list, and repartition
            rdd = rdd.union(namesRdd).partitionBy(2).cache()
 
    return rdd


# Copied this from Thouis's Github.  It takes lines from a graph and converts them into K,V pairs, where K is the page, and V is a list of all the pages that K links to.  
def link_string_to_KV(s):
    src, dests = s.split(': ')                     # split at the colon into source and destinations
    dests = [int(to) for to in dests.split(' ')]   # use a list comprehension to split the destinations at each space
    return (int(src), dests)                       # Return the source and the list of destinations at a tuple

# I copied much of this from Thouis's Github as well.  Trying to make sense of this problem at all...        
if __name__ == "__main__":  
    sc = pyspark.SparkContext()

    #sc.setLogLevel('WARN')

    # Load the pages
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
    page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

    # process links into tuples of node number and a list of all the neighbor node numbers
    neighbor_graph = links.map(link_string_to_KV)
    # partition and cache
    neighbor_graph = neighbor_graph.partitionBy(256).cache()

    # create an RDD for looking up page names from numbers
    # Add 1 in the map to account for 1-indexing as opposed to 0-indexing
    page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
    # Sort them to make them quicker to use later, and cache
    page_names = page_names.sortByKey().cache()

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

    BFS(neighbor_graph, page_names, iterations = 10, inspect = [Kevin_Bacon, Harvard_University], sc)

