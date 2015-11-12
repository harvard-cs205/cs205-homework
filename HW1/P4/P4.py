import P4_bfs
from pyspark import SparkContext, SparkConf
if 'sc' not in globals():
	conf = SparkConf().setAppName('BFS').setMaster('local')
	sc = SparkContext(conf=conf)

if __name__ == "__main__":
	csv=sc.textFile("source.csv")
	num_partitions=8
	data_char=csv.map(lambda x:tuple(x.strip('"').split('","'))).distinct()
	data_comic=data_char.map(P4_bfs.swap_KV)
	graph_edges=data_comic.join(data_comic)\
		.values() \
		.filter(lambda i:i[0]!=i[1]) \
		.distinct()\
		.partitionBy(num_partitions).cache()
	char_list=graph_edges.keys().distinct()
	tot_char=char_list.count()	
	info=''
	start_char=['CAPTAIN AMERICA','MISS THING/MARY','ORWELL']
	
	for char in start_char:
		graph_size=P4_bfs.num_connections_bfs(char,graph_edges,char_list,num_partitions,sc)

		num_nodes=sum(graph_size)	

		rem=tot_char-num_nodes
	 	info=info+char+' has '+str(num_nodes)+' connected nodes.\n'+ str(rem)+\
	 		' of the characters are not connected. \n\n'
 	
 	with open('P4.txt','w') as f:
 		f.write(info)	