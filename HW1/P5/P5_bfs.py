import pyspark
from pyspark import AccumulatorParam

class AccumParam_Set(AccumulatorParam):
    def zero(self, value):
        return value
    def addInPlace(self, value1, value2):
        value1 |= value2
        return value1

def shortest_path_parallel(sc, graph, root_node, dest_node, num_part):
    depth = 0
    queue = set([root_node]) # set to store nodes that will be need to be visited
#     print queue
#     print len(queue)
    traversed_nodes = set() # set to store traversed nodes and its distance from root_node
    # parents_rdd
    parents_rdd = sc.parallelize([(root_node,-1)], num_part)

    while(len(queue) > 0): # while there are more nodes in queue to be visisted
#         print depth
        depth += 1 # increment the depth
        print "depth: ", depth
        # filter on only the nodes in the queue
        filtered_graph = graph.filter(lambda (Node, V): Node in queue)
        #print "filtered_graph:", filtered_graph.values().collect()
        # update the traversed nodes with the nodes in the queue
        traversed_nodes.update(queue)

        # Store parents and child
        parents_child = filtered_graph.flatMapValues(lambda x: x).map(lambda (Parent, Child): (Child, Parent))
        parents_child = parents_child.filter(lambda (Node, V): Node in queue)
        # print "parents_child: ", parents_child.collect()
        # print "traversed_nodes: ", traversed_nodes

        # Union
        parents_rdd = parents_child.subtractByKey(parents_rdd).union(parents_rdd)
        # print "parents_rdd:", parents_rdd.collect()
        

        # Children
        # to store the children of traversed nodes from the queue
        traversed_nodes_children = sc.accumulator(set(), AccumParam_Set()) 
        # Loop through each node in the filtered_graph to add on children
        filtered_graph.values().foreach(lambda x: traversed_nodes_children.add(x))
        
        # print "traversed_nodes_children value:", traversed_nodes_children.value
        # print "traversed_nodes: ", traversed_nodes
        # update queue
        queue = traversed_nodes_children.value - traversed_nodes
        # print "queue: ", queue
        # print "\n"

        if dest_node in traversed_nodes:
            print "found dest_node!"
            # Find shortest path
            lookup = dest_node
            shortest_path = [dest_node]
            while(lookup != -1):
                # print "lookup1: ", lookup
                lookup = parents_rdd.lookup(shortest_path[-1])[0]
                # print "lookup2: ", lookup
                if lookup != -1:
                    shortest_path += [lookup]
            reverse = shortest_path[::-1]
            print [index_name.lookup(index)[0] for index in reverse]
            break
    return len(traversed_nodes) - 1, depth, parents_rdd

def clean_links(line):
    data = line.split(': ')
    data[1] = set(int(x) for x in data[1].split(' '))
    return (int(data[0]), data[1])


if __name__ == '__main__':
    # initiate spark
    sc = pyspark.SparkContext("local", appName="Spark")
    sc.setLogLevel('WARN')

    num_part = 64
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', num_part)
    titles = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', num_part)

    graph = links.map(clean_links, True).cache()
    name_index = titles.zipWithIndex().map(lambda (title, index): (title, index + 1))
    index_name = name_index.map(lambda (title, index): (index, title))
    print "graph: ", graph.take(10)
    print "name_index: ", name_index.take(10)

    root_node = nameindex.lookup('Kevin_Bacon')[0]
    dest_node = nameindex.lookup('Harvard_University')[0]

    print root_node
    print dest_node

    result = shortest_path_parallel(sc, graph, root_node, dest_node, num_part)
    print result

    
