


class BFS(object):
    '''Life is complicated by spark's lazy evaluation. We have to collect at the end of
    every iteration, or progress will be lost! Also, dealing with classes in spark is awful
    due to serialization. So we have to do some serious shennanigans...TERRIBLE'''

    def __init__(self, sc, start_node, network_rdd):

        self.sc = sc
        self.start_node = start_node

        # Cache the network rdd so we don't have to keep recomputing it! It's not changing!
        self.network_rdd = network_rdd.cache()

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

    def run_until_converged(self):
        go = True
        while go:
            before_update = dict(self.collected_distance_rdd)
            self.do_iteration()
            after_update = dict(self.collected_distance_rdd)
            if before_update == after_update:
                go = False
        print 'Finished at end of iteration' , self.cur_iteration - 1 , '!'

    #### STATIC METHODS TO INTERACT WITH SPARK ####
    @staticmethod
    def initialize_distances_static(sc, start_node, network_rdd, cur_iteration):
        network_to_touch = network_rdd.filter(lambda x: x[0] == start_node)
        distance_rdd = network_to_touch.map(lambda x: (x[0], cur_iteration))
        collected_distance_rdd = distance_rdd.collect()

        return collected_distance_rdd

    @staticmethod
    def get_smaller_value(a, b):
        if a < b:
            return a
        else:
            return b


    @staticmethod
    def do_iteration_static(sc, network_rdd, collected_distance_rdd, cur_iteration):
        #TODO: Figure out where to use accumulators...
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
        corrected_distance_rdd = updated_distance_rdd.reduceByKey(BFS.get_smaller_value)

        collected_distance_rdd = corrected_distance_rdd.collect()

        broadcasted_touched.unpersist() # If you don't put this at the end, terrible things happen

        return collected_distance_rdd

class Path_Finder(object):

    num_partitions = 50

    def __init__(self, sc, network_rdd, start_node, end_node):

        self.sc = sc
        # User will define the cache if they want...otherwise computer will melt
        self.network_rdd = network_rdd #Sort by key & use num_partitions beforehand in network_rdd & cache to improve performance
        self.start_node = start_node
        self.end_node = end_node

        # Other helper variables
        self.cur_iteration = 0
        self.collected_distance_rdd = None

        # Run this at the end!
        self.initialize_distances()



    def initialize_distances(self):
        self.collected_distance_rdd = Path_Finder.initialize_distances_static(self.sc,
                                                                      self.start_node,
                                                                      self.network_rdd,
                                                                      self.cur_iteration)
        self.cur_iteration += 1

    def do_iteration(self):
        self.collected_distance_rdd = Path_Finder.do_iteration_static(self.sc,
                                                              self.network_rdd,
                                                              self.collected_distance_rdd,
                                                              self.cur_iteration)
        self.cur_iteration += 1

    def run_until_converged(self):
        go = True
        are_connected = None
        while go:
            print self.cur_iteration
            before_update = dict(self.collected_distance_rdd)
            self.do_iteration()
            after_update = dict(self.collected_distance_rdd)
            if self.end_node in after_update:
                go = False
                are_connected = True
            elif before_update == after_update:
                go = False
                are_connected = False
        if are_connected:
            print 'Nodes are connected!'
        else:
            print 'Nodes do not seem to be connected... :('
        print 'Finished at end of iteration' , self.cur_iteration - 1 , '!'

    #### STATIC METHODS TO INTERACT WITH SPARK ####
    @staticmethod
    def initialize_distances_static(sc, start_node, network_rdd, cur_iteration):
        return [(start_node, (cur_iteration, []))]

    @staticmethod
    def do_iteration_static(sc, network_rdd, collected_distance_rdd, cur_iteration):
        #TODO: Figure out where to use accumulators...
        # The broadcast variable is too big here. Don't use it. It was ok for marvel.

        already_touched = [z[0] for z in collected_distance_rdd]
        already_touched_set = set(already_touched)
        broadcasted_touched = sc.broadcast(already_touched_set)
        network_to_touch = network_rdd.filter(lambda x: x[0] in broadcasted_touched.value)

        old_distance_rdd = sc.parallelize(collected_distance_rdd, Path_Finder.num_partitions)

        # Now do the iteration!
        def get_nodes_to_touch_and_parents(x):
            parent = x[0]
            nodes_to_touch = x[1]
            return [(z, parent) for z in nodes_to_touch]

        nodes_to_touch_and_parents = network_to_touch.flatMap(get_nodes_to_touch_and_parents, preservesPartitioning=True)

        # We now groupby individual
        grouped_by_node = nodes_to_touch_and_parents.groupByKey(numPartitions=Path_Finder.num_partitions)
        grouped_by_node_list = grouped_by_node.map(lambda x: (x[0], list(x[1])), preservesPartitioning=True)
        updated_touched_nodes = grouped_by_node_list.map(lambda x: (x[0], (cur_iteration, x[1])), preservesPartitioning=True)

        updated_distance_rdd = old_distance_rdd.union(updated_touched_nodes)

        def get_smaller_value(a, b):
            '''There are all sorts of subtleties here...but since we don't care about the exact
            path it doesn't matter. Take the parents that are from an earlier iteration.'''
            if a[0] < b[0]:
                return a
            else:
                return b

        corrected_distance_rdd = updated_distance_rdd.reduceByKey(get_smaller_value, numPartitions=Path_Finder.num_partitions)

        collected_distance_rdd = corrected_distance_rdd.collect()

        broadcasted_touched.unpersist()

        return collected_distance_rdd