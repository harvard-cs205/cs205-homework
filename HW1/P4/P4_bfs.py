# this part goes in P4_bfs
# make function taking in graph RDD and char name
# gives shortest path to all other nodes
# diameter <=10


# this part goes in P4_bfs
# make function taking in graph RDD and char name
# gives shortest path to all other nodes
# diameter <=10

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
    print distancesRDD.collect()

    # trying to correct laziness of map
    def myitchecker(children):
        print children
        return lambda x: (x[0],count) if x[0][1] in children else x


    while (not graphRDD.isEmpty() and children != []):
        # update distances
        print count

        newdistancesRDD = distancesRDD.map(myitchecker(children))

        # update graph
        # go through next gen, take out old generation from graphRDD
        #graphRDD = graphRDD.filter(lambda x: x[0] not in parents)
        print newdistancesRDD.collect()

        #children become new parents
        visited += children[:]
        parents = children[:]
        children = []

        # get new children (next gen) in a list
        for i in parents:
            print i
            x = graphRDD.map(lambda x: x).lookup(i)
            if x != []:
                children += x[0]
                print children
        # if no children you are done
        if children == []:
            print "here"
            break
        # make sure no repeats
        children = [j for j in children if j not in visited]


        count += 1

    distancesRDD.collect()
    return newdistancesRDD



