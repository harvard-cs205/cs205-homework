import findspark
findspark.init()
import pyspark
import csv

sc = pyspark.SparkContext()
source_reader = csv.reader(open("source.csv", 'rb'), delimiter=',')

# readuce by key to get RDD with issue -> [super_hero list]
issue_sh = sc.parallelize(list(source_reader), 100).map(lambda x: (x[1].strip(), [x[0].strip()])).reduceByKey(lambda x, y: x + y)

def construct_edges(x):
    i, s_list = x
    l = len(s_list)
    edges = []
    for i in range(l):
        for j in range(l):
            if i != j: 
                edges.append((s_list[i], [s_list[j]]))
    return edges

edges = issue_sh.flatMap(construct_edges)
# might want to broadcast this to all of the partitions 
adj_list = edges.reduceByKey(lambda x, y: x + y)

# need to define a source to initailize v_dist with
source = "ORWELL"
# superhero, (dist, checked_flag, neighbors))
# flatMapper(k):  
# superhero, dist from source
v_dist = sc.parallelize([(source, 0)], 10) 
# initialization values

def closure(dist):
    acc = sc.accumulator(0)

    def map_neighs(x):
        return (x, [dist])
    
    def map_vdist(x):
        n, dists = x 
        if len(dists) > 1: 
            acc.add(1)
        return n, min(dists)

    return acc, map_neighs, map_vdist

acc = sc.accumulator(0)
dist = 1
num_neighs = 1
while (acc.value != num_neighs) and dist < 10:
    acc = sc.accumulator(0)

    def map_vdist(x):
        n, dists = x 
        # vert is already in v_dist 
        if len(dists) > 1: 
            acc.add(1)
        return n, min(dists)

    cur_verts = set(v_dist.filter(lambda x: x[1] == dist - 1).map(lambda x: x[0]).collect())
    neighs = adj_list.filter(lambda x: x[0] in cur_verts).flatMap(lambda x: [(n, dist) for n in x[1]])
    num_neighs = neighs.count()
    v_dist = v_dist.union(neighs).map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x + y).map(map_vdist)
    dist += 1

print v_dist.map(lambda x: x[1]).reduce(lambda x, y: max(x, y))
print acc.value, num_neighs, dist 
1/0

while(acc.value < num_neighs) and dist < 5:
    acc, map_neigh, map_vdist = closure(dist)
    # what can we filter to reduce shuffle operations here?
    cur_verts = v_dist.filter(lambda x: x[1][0] == dist - 1).map(lambda x: (x[0], None))

    # what should we do here instead of this? 
    neighs = cur_verts.join(adj_list).flatMap(lambda x: x[1][1]).map(map_neigh)
      
    #num_neighs = neighs.count()
    v_dist = v_dist.union(neighs).reduceByKey(lambda x, y: x + y).map(map_vdist)
    dist += 1

print acc.value, num_neighs, dist 
print v_dist.collect()

