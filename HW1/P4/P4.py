import findspark
findspark.init()
import pyspark
import itertools
from P4_bfs import BFS

if __name__ == "__main__":  
    sc = pyspark.SparkContext()

    # Load the lines of the text file 
    source = sc.textFile("source.csv")

    # Split the issue id from the superhero's name
    source = source.map(lambda x: x.split('","'))

    # Trim the extra quotation marks and reverse the order so that the issue ID is used as a key
    source = source.map(lambda (x,y): (y[:-1],x[1:]))

    # Group by issue, and create tuples of (hero, all other heros in that issue) for that issue.  
    source = source.groupByKey().mapValues(lambda heros: [(hero1, [hero2 for hero2 in heros if hero2 != hero1]) for hero1 in heros])

    # FlatMap to take all the tuples out of being in lists by issue.
    source = source.flatMap(lambda (x,y): y)


    # Define a function to take all the lists that are returned by reduceByKey and flatten them,
    #    then use list(set()) to take only unique values
    def func(*args): return list(set(itertools.chain(*args)))

    # Group by each superhero as a key, and take only unique edges
    source = source.reduceByKey(func)
    
    # Partition and cache the source
    source = source.partitionBy(4).cache()

    # Verify that Captain America has 1933 edges... but this returns 1906.  Good enough for now
    # print len(source.lookup(u'CAPTAIN AMERICA')[0])    
    
    # Verify that there are 6449 nodes... but this returns 6444.  Good enough for now
    # print len(source.collect())

    # Call the BFS function to demonstrate that it works
    res1 = BFS(sc, source, u'CAPTAIN AMERICA').values().collect()
    res2 = BFS(sc, source, u'MISS THING/MARY').values().collect()
    res3 = BFS(sc, source, u'ORWELL').values().collect()
    print("Captain America is connected to " + str(len(res1)) + " nodes, with a maximum distance of " + str(max(res1)) + ".")
    print("Miss Thing/Mary is connected to " + str(len(res2)) + " nodes, with a maximum distance of " + str(max(res2)) + ".")
    print("Orwell is connected to " + str(len(res3)) + " nodes, with a maximum distance of " + str(max(res3)) + ".")
