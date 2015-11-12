import P5_bfs
from pyspark import SparkContext, SparkConf
if 'sc' not in globals():
	conf = SparkConf().setAppName('BFS').setMaster('local')
	sc = SparkContext(conf=conf)

def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner


if __name__ == "__main__":		
	num_partitions=8
	csv=sc.textFile("source.csv",num_partitions)
	data_char=csv.map(lambda x:tuple(x.strip('"').split('","'))).distinct()
	data_comic=data_char.map(P5_bfs.swap_KV)
	graph_edges=data_comic.join(data_comic)\
		.values() \
		.filter(lambda i:i[0]!=i[1]) \
		.distinct().partitionBy(num_partitions).cache()
	start_char='CAPTAIN AMERICA'
	end_char='CONTONI, PAUL'

	
	result=P5_bfs.find_paths(graph_edges,start_char,end_char,num_partitions,sc)
	graph_size=result[0]
	paths=result[1][0]
	info=''
	for l in paths:
		info=info+'->'.join(l[1])+'\n'
	
	with open('P5_comic_paths.txt','w') as f:
 		f.write(info)