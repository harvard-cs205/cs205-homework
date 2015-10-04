from P4_bfs import *

import findspark
findspark.init('/Users/georgelok/spark')

import pyspark
sc = pyspark.SparkContext(appName="P4")


def lineSplitFunction(line) :
    line = line[1:-1] # Remove unnecessary quotes
    (a,b) = line.split('","') # Split using this delimter since names/issues have ,
    return (b,a)

wlist = sc.textFile('source.csv').map(lineSplitFunction) 

comicGroups = wlist.groupByKey() # This gives us cliques

def flatmapFunctor (x) :
    chars = list(x[1])
    results = []
    for char in chars :
        # Somewhat inefficient, but using remove for some reason was buggy
        results.append((char, set([x for x in chars if x != char])))
    return results

# Assign each character to their clique
allCliques = comicGroups.flatMap(flatmapFunctor) 

# Merge all cliques per character.
allEdges = (allCliques
            .reduceByKey(lambda x1, x2 : x1 | x2)
            .map(lambda x : (x[0], list(x[1]))))

ac1, nodes1 = SSBFS("CAPTAIN AMERICA", allEdges, sc)

ac2, nodes2 = SSBFS("MISS THING/MARY", allEdges, sc)

ac3, nodes3 = SSBFS("ORWELL", allEdges, sc)
print "CAPTAIN AMERICA: " + str(ac1) + " touched nodes"
print "MISS THING/MARY: " + str(ac2) + " touched nodes"
print "ORWELL: " + str(ac3) + " touched nodes"



