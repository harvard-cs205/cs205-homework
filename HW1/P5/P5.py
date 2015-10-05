import pyspark
import time
sc = pyspark.SparkContext()
sc.setLogLevel('WARN')

# Set number of partitions
pc = 100

starttime = time.time()
# Define some functions
def update_dist(v):
    """
    Updates pages' distance from the starting page and makes a list of the previous pages
    in a given page's the shortest path from the starting node.
    """
    if v[0] < 0:
        prev_list = []
        for i in range(1,len(v)):
            prev_list.append(v[i])
        return (counter, prev_list)
    return v
    
def start_dist(k,v):
    " Changes the distance for the starting page to zero. "
    if k == startpage:
        return (k,(0,None))
    else:
        return (k,v)

# Get links-simple-sorted.txt into a list of connected pairs
page_pairs = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
page_pairs  = page_pairs.map(lambda x: x.split(':')).flatMap(
            lambda (k,v): [(int(k),int(i)) for i in v.split()])
page_pairs = page_pairs.partitionBy(pc).cache()

# Find the starting and ending points
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
page_names = page_names.sortByKey().cache()

# Switch 'target' and 'startpage' to find a Harvard -> Kevin path
target = page_names.filter(lambda (k, v): v == 'Kevin_Bacon').collect()[0][0]
startpage = page_names.filter(lambda (k, v): v == 'Harvard_University').collect()[0][0] 

print 'loaded everything at', time.localtime().tm_hour, ':', time.localtime().tm_min
starttime = time.time()
 
# Find the shortest path           
dist = page_pairs.keys().distinct().map(lambda v: (v,-1))
dist = dist.map(lambda (k,v): start_dist(k,v))
dist = dist.partitionBy(pc)
connected = sc.parallelize([(startpage, None)])
connected = connected.partitionBy(pc)
counter = 1
found = False

print 'made it to loop at', time.localtime().tm_hour, ':', time.localtime().tm_min

while not found:
    starttime = time.time()
    connected = page_pairs.join(connected).mapValues(
    lambda v: max(v)).map(lambda (k,v): (v,k)).partitionBy(pc)
    newdists = dist.filter(lambda (k,v): v<0).join(connected)
    newdists = newdists.mapValues(update_dist)
    dist = newdists.union(dist).reduceByKey(
    lambda x,y: max(x,y))
    
    if dist.filter(lambda (k,v): k==target and v>=0).count() > 0:
        found = True
        
    counter += 1
    print 'finished loop', counter-1, 'at', time.localtime().tm_hour, ':', time.localtime().tm_min

# Recover a shortest path    
lenfromstart = -1
length = 0
wherefrom = target
final_path = [target]
while wherefrom != startpage:
    wherefrom = dist.filter(lambda (k,v): k==wherefrom).first()[1][1][0]
    final_path.append(wherefrom)
    length += 1
    
print final_path
final_path_names = []
for i in final_path:
    print i, page_names.filter(lambda (k, v): k == i).collect()[0]
    final_path_names.append(page_names.filter(lambda (k, v): k == i).collect()[0][1])

print 'Final path:\n', final_path_names, '\n\nLength:', length