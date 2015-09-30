class BFS(object):
    '''I figured out how to avoid collecting every iteration...thankfully.'''

    num_partitions = 50

    def __init__(self, sc, start_node, network_rdd):

        self.sc = sc
        self.start_node = start_node

        # Cache the network rdd so we don't have to keep recomputing it! It's not changing!
        self.network_rdd = network_rdd.sortByKey(numPartitions=BFS.num_partitions)

        self.cur_iteration = 0
        self.distance_rdd = None
        self.initialize_distances()


    def initialize_distances(self):
        self.distance_rdd = BFS.initialize_distances_static(self.sc, self.start_node, self.network_rdd,
                                                            self.cur_iteration)
        self.cur_iteration += 1

    def do_iteration(self):
        self.distance_rdd = BFS.do_iteration_static(self.sc, self.network_rdd, self.distance_rdd, self.cur_iteration)
        self.cur_iteration += 1

    def run_until_converged(self):
        go = True
        while go:
            before_update = self.distance_rdd.count()
            self.do_iteration()
            after_update = self.distance_rdd.count()
            if before_update == after_update:
                go = False
        print 'Finished at end of iteration' , self.cur_iteration - 1 , '!'

    #### STATIC METHODS TO INTERACT WITH SPARK ####
    @staticmethod
    def initialize_distances_static(sc, start_node, network_rdd, cur_iteration):
        return sc.parallelize([(start_node, cur_iteration)], BFS.num_partitions)

    @staticmethod
    def get_smaller_value(a, b):
        if a < b:
            return a
        else:
            return b


    @staticmethod
    def do_iteration_static(sc, network_rdd, distance_rdd, cur_iteration):
        #TODO: Figure out where to use accumulators...
        # Pull the needed info out of the network

        joined_network = network_rdd.join(distance_rdd, numPartitions=BFS.num_partitions).coalesce(BFS.num_partitions)
        network_to_touch = joined_network.map(lambda x: (x[0], x[1][0]), preservesPartitioning=True)

        # Now do the iteration!
        nodes_to_touch = network_to_touch.flatMap(lambda x: x[1], preservesPartitioning=True)
        unique_nodes_to_touch = nodes_to_touch.distinct(BFS.num_partitions)
        updated_touched_nodes = unique_nodes_to_touch.map(lambda x: (x, cur_iteration), preservesPartitioning=True)
        updated_distance_rdd = distance_rdd.union(updated_touched_nodes)
        corrected_distance_rdd = updated_distance_rdd.reduceByKey(BFS.get_smaller_value, BFS.num_partitions)

        return corrected_distance_rdd

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
        self.distance_rdd = None

        # Run this at the end!
        self.initialize_distances()



    def initialize_distances(self):
        self.distance_rdd = Path_Finder.initialize_distances_static(self.sc,
                                                                      self.start_node,
                                                                      self.network_rdd,
                                                                      self.cur_iteration)
        self.cur_iteration += 1

    def do_iteration(self):
        self.distance_rdd = Path_Finder.do_iteration_static(self.sc,
                                                              self.network_rdd,
                                                              self.distance_rdd,
                                                              self.cur_iteration)
        self.cur_iteration += 1

    def run_until_converged(self):
        go = True
        are_connected = None
        while go:
            print self.cur_iteration
            before_update = dict(self.distance_rdd)
            self.do_iteration()
            after_update = dict(self.distance_rdd)
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
        return sc.parallelize([(start_node, (cur_iteration, []))], Path_Finder.num_partitions)

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




class Connected_Components(object):
    num_partitions = 50

    def __init__(self, sc, network_rdd):

        self.sc = sc
        # User will define the cache if they want...otherwise computer will melt
        self.network_rdd = network_rdd #Sort by key & use num_partitions beforehand in network_rdd & cache to improve performance
        self.connected_collected_rdd = None

        # Other helper variables
        self.cur_iteration = 0

        # Run this at the end!
        self.initialize_groups()

    def initialize_groups(self):
        self.connected_collected_rdd = Connected_Components.initialize_groups_static(self.sc, self.network_rdd)
        self.cur_iteration += 1

    def do_iteration(self):
        self.connected_collected_rdd = Connected_Components.do_iteration_static(self.sc, self.connected_collected_rdd)
        self.cur_iteration += 1

    def get_num_unique_groups(self):
        list_of_indices = map(lambda x: x[1][1], self.connected_collected_rdd)
        return len(set(list_of_indices))

    def run_until_converged(self):
        go = True
        groups_before_update = None
        groups_after_update = None
        while go:
            print 'Beginning of Iteration', self.cur_iteration
            groups_before_update = self.get_num_unique_groups()
            print 'Before update:' , groups_before_update, 'groups'
            self.do_iteration()
            groups_after_update = self.get_num_unique_groups()
            print 'After update:' , groups_after_update , 'groups'
            if groups_before_update == groups_after_update:
                go = False
        print 'Finished at end of iteration' , self.cur_iteration - 1 , '!'
        print 'There are ', groups_after_update, ' groups in the network.'

    #### STATIC METHODS TO INTERACT WITH SPARK ####
    @staticmethod
    def initialize_groups_static(sc, network_rdd):
        return network_rdd.zipWithIndex().map(lambda x: (x[0][0], (x[0][1], x[1])), preservesPartitioning=True).collect()


    @staticmethod
    def do_iteration_static(sc, connected_collected_rdd):
        connected_rdd = sc.parallelize(connected_collected_rdd, Connected_Components.num_partitions)

        def get_parent_index(x):
            parents = x[1][0]
            index = x[1][1]
            return [(z, index) for z in parents]

        def get_smaller_index(a, b):
            if a < b:
                return a
            return b

        parent_indexes = connected_rdd.flatMap(get_parent_index, preservesPartitioning=True)
        parent_with_smallest_index = parent_indexes.reduceByKey(get_smaller_index, numPartitions=Connected_Components.num_partitions)
        # We now have to join to the connected_rdd...not sure if a join or a broadcast variable is faster here.
        new_connected_rdd = connected_rdd.join(parent_with_smallest_index, numPartitions=Connected_Components.num_partitions)
        connected_rdd = new_connected_rdd.map(lambda x: (x[0], (x[1][0][0], x[1][1])))
        return connected_rdd.collect()