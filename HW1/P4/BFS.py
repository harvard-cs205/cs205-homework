
def get_smaller_value(a, b):
    if a < b:
        return a
    else:
        return b

class BFS(object):
    '''Life is complicated by spark's lazy evaluation. We have to collect at the end of
    every iteration, or progress will be lost!'''

    def __init__(self, sc, start_node, network_rdd):
        self.sc = sc
        self.start_node = start_node
        self.network_rdd = network_rdd
        self.cur_iteration = 0
        self.collected_distance_rdd = None

    def initialize_distances(self):
        network_to_touch = self.network_rdd.filter(lambda x: x[0] == self.start_node)
        distance_rdd = network_to_touch.map(lambda x: (x[0], self.cur_iteration))
        self.collected_distance_rdd = distance_rdd.collect()
        self.cur_iteration = 1


    def do_iteration(self):
        # Pull the needed info out of the network
        distance_rdd = self.sc.parallelize(self.collected_distance_rdd)

        network_to_touch = self.network_rdd.filter(lambda x: x[0] == self.start_node)
        distance_rdd = network_to_touch.map(lambda x: (x[0], self.cur_iteration))
        nodes_to_touch = network_to_touch.flatMap(lambda x: x[1])
        unique_nodes_to_touch = nodes_to_touch.distinct()
        updated_touched_nodes = unique_nodes_to_touch.map(lambda x: (x, 1))
        updated_distance_rdd = distance_rdd.union(updated_touched_nodes)
        corrected_distance_rdd = updated_distance_rdd.reduceByKey(get_smaller_value)