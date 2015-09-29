import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

def swap_KV(KV):
	return (KV[1],KV[0])

if __name__ == "__main__":
	csv=sc.textFile("source.csv")
	data_char=csv.map(lambda x:tuple(x.strip('"').split('","'))).distinct()
	data_comic=data_char.map(swap_KV)
	graph_nodes=data_comic.join(data_comic)\
		.values() \
		.filter(lambda i:i[0]!=i[1]) \
		.distinct().repartition(8).cache()
	tot_char=data_char.keys().distinct().count()	
	
	info=''
	start_char=['CAPTAIN AMERICA']#,'MISS THING/MARY','ORWELL']
	for char in start_char:
		graph=sc.parallelize([(char,0)])
		graph_level=graph
		level=0
		Nmax=1
		graph_size=[1]
		
		#graph_nodes_run_new=graph_nodes.subtractByKey(graph_level)
		#graph_nodes_run=graph_nodes_run_new.map(swap_KV).cache()
		
# 		for level in range(Nmax):
 		while graph_size[level]!=0:
 			accum = sc.accumulator(0)
 			graph_level=graph_level.join(graph_nodes)\
 				.map(lambda i:(i[1][1],level+1)).distinct().subtractByKey(graph)\
				.coalesce(8).cache()
			
 			graph=graph+graph_level
 			#graph_level.foreach(lambda i:accum.add(1))
# 			# graph_nodes_run=graph_nodes_run.subtractByKey(graph_level).map(swap_KV)\
# # 				.subtractByKey(graph_level_new).map(swap_KV).cache()
# # 			graph_level=graph_level_new			
 			graph=graph+graph_level
 			graph_size.append(graph_level.count())
 			level+=1
#		num_nodes=graph.count()
# 		rem=tot_char-num_nodes
# 		info=info+char+' has '+str(num_nodes)+' connected nodes.\n'+ str(rem)+\
# 			' of the characters are not connected. \n\n'
# 	
# 	with open('P4.txt','w') as f:
# 		f.write(info)
# # 
# # 	