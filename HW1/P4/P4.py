#setup spark
import findspark
findspark.init('/home/raphael/spark')

import pyspark
sc = pyspark.SparkContext(appName="myAppName")

#call the bfs function
from P4_bfs import distance


#LOAD DATA and CREATE THE GRAPH RDD
c_i = sc.textFile("/home/raphael/Downloads/source.csv",minPartitions=23,use_unicode=True)
#put the magazine as key + string manipulation
i_c = c_i.map(lambda line:(line.split('"')[3],line.split('"')[1]))
#group the heroes that have been together
i_gc = i_c.groupByKey().mapValues(list)
#key: hero, value: other heroes with him per magazine
gc = i_gc.flatMap(lambda (k,v): [(d, list(set(v).difference(set([d])))) for d in v])
#key: hero, value: another hero that has been in the same magazine
gc_final = gc.flatMap(lambda (k,v): [(k,b) for b in v])
#key: hero, value: list of all the heroes that have appeared in the same magazine
gc_final = gc_final.groupByKey().mapValues(list)
#create a partition
gc_final.partitionBy(23).cache()
#force to recompute
gc_final.foreach(lambda x: x)

#this helper function is a min to order the elements in function of the 
#distance so that the first taken is of the largest distance for a visited node
def order(k,v):
    import sys
    if v[0]==sys.maxint:
        return 0
    else:
        return -v[0]

#Running the code for asked result
def run(root):
    """
    This is an helper function to get the results asked
    """
    #call the distance function for "Captain America
    db, t = distance(sc, gc_final, root)
    print "For ,", root

    #get the number of nodes in the graph by counting the visited items
    print "the number of nodes in the graph is ", db.filter(lambda (k,v): v[2]!=0).count()

    #the first element of the value in the key value pair is the diameter
    print "the diameter is ", db.takeOrdered(1,lambda (k,v):order(k,v))[0][1][0]
    return 0


roots = [u"CAPTAIN AMERICA",u"MISS THING/MARY", u"ORWELL"]
for root in roots:
    run(root)
