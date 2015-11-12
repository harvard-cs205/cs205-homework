from pyspark import SparkContext, SparkConf
import P5_bfs

if 'sc' not in globals():
	conf = SparkConf().setAppName('BFS')
	sc = SparkContext(conf=conf)

if __name__ == "__main__":
	numPartitions=100		
	sc.setLogLevel("ERROR")
	links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt',numPartitions)
	page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt',numPartitions)
	page_names = page_names.zipWithIndex().map(lambda (K,V):(V+1,K))\
		.partitionBy(numPartitions)
	graph_edges=links.map(lambda i:tuple(i.split(': '))) \
		.mapValues(lambda i:i.split(' ')) \
		.flatMapValues(lambda i:i).map(lambda i:(int(i[0]),int(i[1])))\
		.partitionBy(numPartitions).cache()
	start_page='Kevin_Bacon'
	end_page='Harvard_University'
	
	result=P5_bfs.find_paths(graph_edges,start_page,end_page,page_names,numPartitions,sc)

	graph_size=result[0]
	char_paths=result[1][0]
	info=''
	for l in paths:
		info=info+' -> '.join(l)+'\n'
	
	with open('P5_wiki_paths.txt','w') as f:
 		f.write(info)