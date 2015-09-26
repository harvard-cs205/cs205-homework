'''Assumes that you have a variable called sc that is your spark context! Won't work otherwise.
The issue is that the context cannot be passed in, weirdly...'''

def get_smaller_value(a, b):
    if a < b:
        return a
    else:
        return b

class BFS(object):
    '''Life is complicated by spark's lazy evaluation. We have to collect at the end of
    every iteration, or progress will be lost! Also, dealing with classes in spark is awful
    due to serialization. So we have to do some serious shennanigans...TERRIBLE'''

    def __init__(self, sc, start_node, network_rdd):

        self.sc = sc
        self.start_node = start_node
        self.network_rdd = network_rdd
        self.cur_iteration = 0
        self.collected_distance_rdd = None
        self.initialize_distances()


    def initialize_distances(self):
        self.collected_distance_rdd = BFS.initialize_distances_static(self.sc,
                                                                      self.start_node,
                                                                      self.network_rdd,
                                                                      self.cur_iteration)
        self.cur_iteration += 1

    def do_iteration(self):
        self.collected_distance_rdd = BFS.do_iteration_static(self.sc,
                                                              self.network_rdd,
                                                              self.collected_distance_rdd,
                                                              self.cur_iteration)
        self.cur_iteration += 1

    #### STATIC METHODS TO INTERACT WITH SPARK ####
    @staticmethod
    def initialize_distances_static(sc, start_node, network_rdd, cur_iteration):
        network_to_touch = network_rdd.filter(lambda x: x[0] == start_node)
        distance_rdd = network_to_touch.map(lambda x: (x[0], cur_iteration))
        collected_distance_rdd = distance_rdd.collect()

        return collected_distance_rdd

    @staticmethod
    def do_iteration_static(sc, network_rdd, collected_distance_rdd, cur_iteration):
        # Pull the needed info out of the network
        already_touched = [z[0] for z in collected_distance_rdd]
        already_touched_set = set(already_touched)
        broadcasted_touched = sc.broadcast(already_touched_set)
        network_to_touch = network_rdd.filter(lambda x: x[0] in broadcasted_touched.value)

        old_distance_rdd = sc.parallelize(collected_distance_rdd)


        # Now do the iteration!
        nodes_to_touch = network_to_touch.flatMap(lambda x: x[1])
        unique_nodes_to_touch = nodes_to_touch.distinct()
        updated_touched_nodes = unique_nodes_to_touch.map(lambda x: (x, cur_iteration))
        updated_distance_rdd = old_distance_rdd.union(updated_touched_nodes)
        corrected_distance_rdd = updated_distance_rdd.reduceByKey(get_smaller_value)

        collected_distance_rdd = corrected_distance_rdd.collect()

        broadcasted_touched.unpersist() # If you don't put this at the end, terrible things happen

        return collected_distance_rdd