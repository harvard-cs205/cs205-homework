from P4 import *
import copy

# this is general function to return the number of nodes each step touched
def number(key):
    keys=copy.copy(key)
    keys_backup=copy.copy(key)
    for i in range(5):
        keys+=final.filter(lambda x: x[0] in key).flatMap(lambda x: x[1]).distinct().collect()
        key=set(keys)-set(keys_backup)
        keys_backup=set(keys)
        print len(key)  

# print the number of nodes touched in every iteration
number([u'CAPTAIN AMERICA'])
number([u'MISS THING/MARY'])
number([u'ORWELL'])

# this is bfs function, which takes key(character name) and rdd(the graph) as input, and outputs the number of shortest steps
def bfs(key,rdd):
    keys=copy.copy(key)
    keys_backup=copy.copy(key)
    i=0
    while len(key)!=0:
        print ("Step ""%d" %i+ ": find ""%d" %len(key) +" nodes")
        keys+=rdd.filter(lambda x: x[0] in key).flatMap(lambda x: x[1]).distinct().collect()
        key=set(keys)-set(keys_backup)
        keys_backup=set(keys)
        i+=1
    print("In all, we search " "%d" %(i-1) +" steps")


# this is a test:
bfs([u'CAPTAIN AMERICA'],final)
