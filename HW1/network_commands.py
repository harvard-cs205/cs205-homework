import numpy as np

num_partitions = 40

class BFS(object):
    '''I figured out how to avoid collecting every iteration...thankfully.'''
    def __init__(self, sc, start_node, network_rdd):

        self.sc = sc
        self.start_node = start_node

        # Cache the network rdd so we don't have to keep recomputing it! It's not changing!
        self.network_rdd = network_rdd.partitionBy(num_partitions).cache()

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
        return sc.parallelize([(start_node, cur_iteration)], num_partitions).partitionBy(num_partitions)

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

        joined_network = network_rdd.join(distance_rdd)
        network_to_touch = joined_network.map(lambda x: (x[0], x[1][0]), preservesPartitioning=True)

        # Now do the iteration!
        nodes_to_touch = network_to_touch.flatMap(lambda x: x[1])
        unique_nodes_to_touch = nodes_to_touch.distinct(num_partitions)
        updated_touched_nodes = unique_nodes_to_touch.map(lambda x: (x, cur_iteration), preservesPartitioning=True)
        updated_distance_rdd = distance_rdd.union(updated_touched_nodes)
        corrected_distance_rdd = updated_distance_rdd.reduceByKey(BFS.get_smaller_value, num_partitions) # Does the partitioning for us

        return corrected_distance_rdd.cache()

class Path_Finder(object):

    def __init__(self, sc, network_rdd, start_node, end_node):

        self.sc = sc
        # User will define the cache if they want...otherwise computer will melt
        #Sort by key & use num_partitions beforehand in network_rdd & cache to improve performance
        self.network_rdd = network_rdd.partitionBy(num_partitions).cache()
        self.start_node = start_node
        self.end_node = end_node

        # Other helper variables
        self.cur_iteration = 0
        self.distance_rdd = None

        # Run this at the end!
        self.initialize_distances()

    def initialize_distances(self):
        self.distance_rdd = Path_Finder.initialize_distances_static(self.sc, self.start_node, self.network_rdd, self.cur_iteration)
        self.cur_iteration += 1

    def do_iteration(self):
        self.distance_rdd = Path_Finder.do_iteration_static(self.sc, self.network_rdd, self.distance_rdd, self.cur_iteration)
        self.cur_iteration += 1

    def run_until_converged(self):
        go = True
        are_connected = None
        while go:
            print self.cur_iteration
            before_update = self.distance_rdd.count()
            self.do_iteration()
            after_update = self.distance_rdd.count()
            if len(self.distance_rdd.lookup(self.end_node)) != 0:
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

    def get_random_path(self):
        '''Once the solution has converged, get a random path from start to finish.'''
        chosen_parent = self.end_node
        path_back = [self.end_node]

        go = True
        while go:
            potential_parents = self.distance_rdd.lookup(chosen_parent)[0][1]
            # Let's make the parent choice random to get variability
            chosen_parent = np.random.choice(potential_parents)
            path_back.append(chosen_parent)
            if chosen_parent == self.start_node:
                go = False
        path_forwards = list(reversed(path_back))
        return path_forwards

    #### STATIC METHODS TO INTERACT WITH SPARK ####
    @staticmethod
    def initialize_distances_static(sc, start_node, network_rdd, cur_iteration):
        return sc.parallelize([(start_node, (cur_iteration, []))]).partitionBy(num_partitions)

    @staticmethod
    def do_iteration_static(sc, network_rdd, distance_rdd, cur_iteration):
        joined_network = distance_rdd.join(network_rdd)
        network_to_touch = joined_network.map(lambda x: (x[0], x[1][1]), preservesPartitioning=True)

        # Now do the iteration!
        def get_nodes_to_touch_and_parents(x):
            parent = x[0]
            nodes_to_touch = x[1]
            return [(z, parent) for z in nodes_to_touch]

        nodes_to_touch_and_parents = network_to_touch.flatMap(get_nodes_to_touch_and_parents)

        # We now groupby individual
        grouped_by_node = nodes_to_touch_and_parents.groupByKey()
        grouped_by_node_list = grouped_by_node.map(lambda x: (x[0], list(x[1])), preservesPartitioning=True)
        updated_touched_nodes = grouped_by_node_list.map(lambda x: (x[0], (cur_iteration, x[1])), preservesPartitioning=True)

        updated_distance_rdd = distance_rdd.union(updated_touched_nodes)

        def get_smaller_value(a, b):
            '''There are all sorts of subtleties here...but since we don't care about the exact
            path it doesn't matter. Take the parents that are from an earlier iteration.'''
            if a[0] < b[0]:
                return a
            else:
                return b

        corrected_distance_rdd = updated_distance_rdd.reduceByKey(get_smaller_value, num_partitions) # This does partitioning for us!

        return corrected_distance_rdd.cache()

class Connected_Components(object):
    '''The network rdd *has* to be reversed now! i.e. The structure must be (node, [parents]), not (node, [children])'''

    def __init__(self, sc, network_rdd):

        self.sc = sc
        # User will define the cache if they want...otherwise computer will melt
        self.network_rdd = network_rdd # ironically we only need this once
        self.connected_rdd = None

        # Other helper variables
        self.cur_iteration = 0

        self.num_unique_vs_iteration = None

        # Run this at the end!
        self.initialize_groups()

    def initialize_groups(self):
        self.connected_rdd = Connected_Components.initialize_groups_static(self.sc, self.network_rdd)
        self.cur_iteration += 1

    def do_iteration(self):
        self.connected_rdd = Connected_Components.do_iteration_static(self.sc, self.connected_rdd)
        self.cur_iteration += 1

    def get_num_unique_groups(self):
        list_of_indices = self.connected_rdd.map(lambda x: x[1][1])
        num_distinct_indices = list_of_indices.distinct().count()
        return num_distinct_indices

    def run_until_converged(self):
        go = True

        self.num_unique_vs_iteration = []

        previous_num_groups = self.get_num_unique_groups()
        groups_before_update = previous_num_groups

        self.num_unique_vs_iteration.append(groups_before_update)

        while go:
            print 'Beginning of Iteration', self.cur_iteration
            print 'Before update:' , groups_before_update, 'groups'
            self.do_iteration()
            groups_before_update = self.get_num_unique_groups()
            self.num_unique_vs_iteration.append(groups_before_update)

            if groups_before_update == previous_num_groups:
                go = False
            else:
                previous_num_groups = groups_before_update
        print 'Finished at end of iteration' , self.cur_iteration - 1 , '!'
        print 'There are ', groups_before_update, ' groups in the network.'

    #### STATIC METHODS TO INTERACT WITH SPARK ####
    @staticmethod
    def initialize_groups_static(sc, network_rdd):
        labeled_network = network_rdd.zipWithIndex().map(lambda x: (x[0][0], (x[0][1], x[1])))
        return labeled_network.partitionBy(num_partitions).cache()


    @staticmethod
    def do_iteration_static(sc, connected_rdd):

        def get_parent_index(x):
            parents = x[1][0]
            index = x[1][1]
            return [(z, index) for z in parents]

        def get_smaller_index(a, b):
            if a < b:
                return a
            return b

        parent_indexes = connected_rdd.flatMap(get_parent_index)
        parent_with_smallest_index = parent_indexes.reduceByKey(get_smaller_index, num_partitions) # This is partitioned
        joined_rdd= connected_rdd.join(parent_with_smallest_index)
        connected_rdd = joined_rdd.map(lambda x: (x[0], (x[1][0][0], x[1][1])), preservesPartitioning=True) # Should preserve partitioning...
        return connected_rdd.cache()