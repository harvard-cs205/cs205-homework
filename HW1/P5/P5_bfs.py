import pyspark
sc = pyspark.SparkContext()
sc.setLogLevel('WARN')

def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)


def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner


def ssbfs(neighbors,page_names,inspect=[]):
	neighbors=neighbors.cache()

    originPage=inspect[0] # Passing in the node ID's for the origin
    endPage=inspect[1] # Passing in the node ID for the goal end page

 	Found=False

	newNeighbors=neighbors.filter(lambda (k,v): k == originPage).flatMapValues(lambda x: x)
	newNeighborsList=newNeighbors.map(lambda (k,v): v).collect()
	Found=endPage in rightNeighborsList

	oldNeighborsList=newNeighborsList

	allNeighbors_touched=[]
	allNeighbors_touched.append(newNeighborsList)
	ii=1
    while Found is False:
    	ii=ii+1
    	newNeighbors=neighbors.filter(lambda (k,v): k in oldNeighborsList).flatMapValues(lambda x: x)
		newNeighborsList=newNeighbors.map(lambda (k,v): v).collect()
		allNeighbors_touched.append(newNeighborsList)
		Found= endPage in newNeighborsList

		oldNeighborsList=newNeighborsList

	# NOW RUN THE ALGORITHM TO FIND THE PATH IN THE SHORTEST CONNECTION
	find_path(neighbors,page_names,inpect,allNeighbors_touched,ii)

def find_path(neighbors,page_names,inspect,allNeighbors_touched,ii):
	document = open('P5.txt','w')
	neighbors=neighbors.cache()

    originPage=inspect[0] # Passing in the node ID's for the origin
    endPage=inspect[1] # Passing in the node ID for the goal end page

    llist=[endpage]

    nextNode=neighbors.lookup(endpage)[0][1]
    for i in range(ii):
    	llist.append(nextNode)
    	tt=neighbors.lookup(nextNode)
    	temp_node=nextNode
    llist_name =[]
    for i in reversed(llist):
    	temp_node=neighbors.lookup(i)[0]
    	llist_name.append(temp_node)
    
    printResultstoFile(document, inspect,llist)


def printResultstoFile(document, inpsect,llist):
	document.write('From '+str(inspect[0])+' to '+str(inspect[1])+' it took '+str(len(llist))+' steps.')
	document.write('\n')
	document.write('The path it took was the following:')
	string=''
	for ii in range(len(llist)):
		string=string,str(llist[ii]),' to ',
	document.write(string)
	document.write('\n')

##### BEGIN REAL CODE #########

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
#links = sc.textFile('links-simple-sorted.txt')
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)
#page_names = sc.textFile('titles-sorted.txt')

# process links into (node #, [neighbor node #, neighbor node #, ...]
neighbor_graph = links.map(link_string_to_KV)

# create an RDD for looking up page names from numbers
# remember that it's all 1-indexed
page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
page_names = page_names.sortByKey().cache()

#######################################################################
# set up partitioning - we have roughly 16 workers, if we're on AWS with 4
# nodes not counting the driver.  This is 16 partitions per worker.
#
# Cache this result, so we don't recompute the link_string_to_KV() each time.
#######################################################################
neighbor_graph = neighbor_graph.partitionBy(256).cache()

# find Kevin Bacon
Kevin_Bacon = page_names.filter(lambda (K, V): V == 'Kevin_Bacon').collect()
# This should be [(node_id, 'Kevin_Bacon')]
assert len(Kevin_Bacon) == 1
Kevin_Bacon = Kevin_Bacon[0][0]  # extract node id

# find Harvard University
Harvard_University = page_names.filter(lambda (K, V):
                                       V == 'Harvard_University').collect()
# This should be [(node_id, 'Harvard_University')]
assert len(Harvard_University) == 1
Harvard_University = Harvard_University[0][0]  # extract node id

ssbfs(neighbor_graph, page_names, inspect=[Kevin_Bacon, Harvard_University])

ssbfs(neighbor_graph, page_names, inspect=[Harvard_University, Kevin_Bacon])

# INSTRUCTION FOR LOADING IT ON THE CLUSTER

# Use SFTP with the mkey file loaded as the public key, then push file into repo

#  aws emr create-cluster --name "Spark cluster" --release-label emr-4.1.0 --applications Name=Spark --ec2-attributes KeyName=SparkKeyPair --instance-type m3.xlarge --instance-count 5 --use-default-roles

# spark-submit --num-executors 4 --executor-cores 4 --executor-memory 8g PageRank.py
