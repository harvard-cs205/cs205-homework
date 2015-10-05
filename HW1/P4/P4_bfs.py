import findspark
findspark.init('/Users/Grace/spark-1.5.0-bin-hadoop2.6/')
import pyspark
import copy

MAX_DIAMETER = 10


def BFS_with_diameter(graph, root_node):
    queue = [root_node]
    visited_nodes = dict()

    for distance in range(0, MAX_DIAMETER+1):
        neighbor = []
        for node in queue:
            if node not in visited_nodes:
                visited_nodes[node] = distance
                neighbor = neighbor + graph.lookup(node)[0]
        queue = neighbor #move to next level-distance

    print "Root node : ", root_node
    print "Number of nodes touched : ", len(visited_nodes)-1
    return visited_nodes


#print BFS_with_diameter(marvel_character_graph, 'CAPTAIN AMERICA') #too slow..;;;
#print BFS_with_diameter(marvel_character_graph, 'ORWELL')
#print BFS_with_diameter(marvel_character_graph, 'MISS THING/MARY')
################################

INFINITY = 1000


#update_distance(('a', (1, ['a'])))
def update_distance(level, queue):
    return lambda (node, (distance, neighbors)): (node, (min(distance, level+1), neighbors)) if node in queue else (node, (distance, neighbors))


def BFS_relax_diameter(graph, root_node, sc):
    # dist_graph : RDD of (node, (distance, [neighbors]))
    # initialize root_node:0, else:INFINITY
    dist_graph = graph.map(lambda (node, neighbors): (node, (0, neighbors)) if (node == root_node) else (node, (INFINITY, neighbors)))
    accum = sc.accumulator(0)
    #level = 0

    while True:
        #print 'level', level
        #print dist_graph.map(lambda (node, (distance, neighbors)): (node, distance)).sortBy(lambda (node, distance): distance).take(10)
        # queue nodes whose distance == level
        level = accum.value
        queue_rdd = dist_graph.filter(lambda (node, (distance, neighbors)): distance == level)
        #print 'while start'
        #print queue_rdd.count() #<- this is really useful to debug how many nodes touched through each level

        if queue_rdd.count() == 0 :
            break
        else :
            queue_rdd = queue_rdd.map(lambda (node, (distance, neighbors)): neighbors)
            queue = queue_rdd.reduce(lambda x, y : list(set(x + y)))
            #print len(queue), level
            #print 'the moment when i use level', level, level+1
            #dist_graph = dist_graph.map(lambda (node, (distance, neighbors)): (node, (min(distance, copy.deepcopy(level)+1), neighbors)) if node in queue
            #                                                                   else (node, (distance, neighbors)))
            dist_graph = dist_graph.map(update_distance(level, queue))
            #print 'after update'
            #print dist_graph.map(lambda (node, (distance, neighbors)): (node, distance)).sortBy(lambda (node, distance): distance).take(10)
            #print dist_graph.count()

            accum.add(1)
            #level+=1
            #print 'after acuum update'
            #print accum.value
            #print dist_graph.map(lambda (node, (distance, neighbors)): (node, distance)).sortBy(lambda (node, distance): distance).take(10)


    #print 'after while'
    #print dist_graph.count()
    filtered_graph = dist_graph.filter(lambda (node, (distance, neighbors)): distance != INFINITY)
    #print filtered_graph.count()
    filtered_graph = filtered_graph.map(lambda (node, (distance, neighbors)): (node, distance))
    filtered_graph = filtered_graph.sortBy(lambda (node, distance): distance, ascending=False)
    print "Root node : ", root_node
    print "Number of nodes touched : ", filtered_graph.count()-1
    print "diameter : ", accum.value-1
    return filtered_graph.collect()
