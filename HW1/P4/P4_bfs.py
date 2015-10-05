
import pyspark
from P4 import*


# root key
key = ["ORWELL"]

# newly-found number of nodes in this layer 
#count = len(key)
# all keys so far
l = ["ORWELL"] 
accum = sc.accumulator(0)
accum.add(1)

while accum.value != 0:
#for i in range(5):
    accum.value = 0
    # get the nodes of next layer
    k = rdd_g.filter(lambda x: x[0] in key).flatMap(lambda r: r[1]).collect()
    
    # subtract previous nodes to get the newly-found nodes in this layer
    key = list(set(k) - set(l))
    
    # newly-found number of nodes in this layer 
    count = len(key)
    print count
    
    if count != 0:
        accum.add(1)
        
    # all nodes searched so far
    l = l + key
    









