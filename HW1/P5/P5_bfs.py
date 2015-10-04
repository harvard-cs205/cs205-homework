from pyspark import SparkContext
import itertools
from P5_helper import *
sc = SparkContext("local", "Simple App")

[2729536, 3554817, 3554854, 2152782]
"National_Lampoon's_Animal_House"
'National_Lampoon_(magazine)'
graph = sc.parallelize([(1,(0, None, [2,3])), (2, (float("inf"), None, [5,3])), (3, (float("inf"), None, [5,1,10])), (10, (float("inf"), None, [2])), (5, (float("inf"), None, []))])

print search_bfs_new(1,10, graph, sc)

#links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
neighbor_graph = links.map(construct_node(2152782))
neighbor_graph = neighbor_graph.partitionBy(256).cache()
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)
page_names = page_names.zipWithIndex().map(lambda (n, id) : (id+1, n))
page_names = page_names.sortByKey().cache()


# page_names.filter(lambda x: x == "Harvard_University").take(1)
def construct_node(start):
	def helper(line):
		src, dests = line.split(': ')
		dests = [int(to) for to in dests.split(' ')]
		if int(src) == start:
			return (int(src), (0, None, dests))
		else:
			return (int(src), (float("inf"), None, dests))
	return helper

# def find_index(page_names, input):
# 	i = 1
# 	while True:
# 		if page_names[i-1] == input:
# 			print i
# 			break
# 		i += 1

def get_name(page_names, index):
	return page_names[index-1]

# # links = links.map(lambda x: ((int(x.split()[0][:-1])), (x.split()[1:])))
# harvard = links.filter(lambda x: (int(x.split()[0][:-1]) == 2152782))

# Harvard = find_index(my_list, "Harvard_University") #2152782
# Kevin_Bacon = find_index(my_list, "Kevin_Bacon") #2729536

print search_bfs_new(2152782, 2729536, neighbor_graph, sc)


# # aws emr create-cluster --name "Spark cluster" --release-label emr-4.1.0 --applications Name=Spark --ec2-attributes KeyName=Vivek --instance-type m3.xlarge --instance-count 3 --use-default-roles
# # j-2RM4I945XUFZ6
# aws emr describe-cluster --cluster-id j-1CWQV92F9ES6W

# aws emr terminate-clusters --cluster-ids j-25MLW8OHMGYD4

# ~/Documents/JuniorYear/CS205/spark-1.5.0/bin/pyspark