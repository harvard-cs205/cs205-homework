# this part goes in P4_bfs
# make function taking in graph RDD and char name
# gives shortest path to all other nodes
# diameter <=10


# initiallizing spark
from pyspark import SparkContext, SparkConf
sc = SparkContext(conf=SparkConf())
sc.setLogLevel("ERROR")



# outside calculations because of lazy RDD updating
def updatedist(children,visited,count):
    print count
    children = [j for j in children if j not in visited]
    return lambda x: (x[0],count) if x[0][1] in children else x


# x is [visited,parents,children]
def updatefamily(x,graphRDD):
    x[2] = [j for j in x[2] if j not in x[0]]
    x[0] += x[2][:]
    x[1] = x[2][:]
    x[2] = []

    # go through next gen, take out old generation from graphRDD
    graphRDD = graphRDD.filter(lambda x: x[0] not in parents)

    return [x[0],x[1],x[2]]

def ssbfs(graphRDD,name):
    distancesRDD = graphRDD.keys().distinct().map(lambda x: ((name,x),0))
    newdistancesRDD = sc.parallelize([])

    count = 1

    visited = [name]
    parents = [name]
    children = []
    x = graphRDD.map(lambda x: x).lookup(name)
    if x != []:
        children += x[0]

    # trying to correct laziness of map


    while (not graphRDD.isEmpty() and children != []):
        # update distances
        distancesRDD = distancesRDD.map(updatedist(children,visited,count)).partitionBy(10)


        #children become new parents, optimize graphRDD
        [visited,parents,children] = updatefamily([visited,parents,children],graphRDD)

        # get new children (next gen) in a list
        for i in parents:
            x = graphRDD.map(lambda x: x).lookup(i)
            if x != []:
                children = x[0]

        # if no children you are done
        if children == []:
            break

        count += 1

    return distancesRDD
