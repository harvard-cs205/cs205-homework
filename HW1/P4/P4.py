import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from itertools import chain



if __name__ == "__main__":
	csv=sc.textFile("source.csv")
	data_char=csv.map(lambda x:tuple(x.strip('"').split('","'))).distinct().cache()
	data_char_next=data_char
	data_comic=data_char.map(lambda i: (i[1],i[0]))
	data_comic_next=data_comic
	#grouped_by_name=data.groupByKey().map(lambda i:(i[0],list(set(i[1]))))
	#grouped_by_comic=data.map(lambda i: (i[1],i[0])) \
	#	.groupByKey().map(lambda i:(i[0],list(set(i[1]))))
	graph=sc.parallelize([('CAPTAIN AMERICA',0)])
	graph_level=graph
	level=0
	Nmax=5
	graph_size=[1]
	
	#graph_level_arr=[['CAPTAIN AMERICA']]
	for level in range(0,Nmax):	
		#get list of comics for which character was present
		comic_list=graph_level.join(data_char) \
			.flatMap(lambda i:i[1][1:]).distinct()
		#get list of characters from those comics	
		char_list=comic_list.map(lambda i:(i,1)) \
			.join(data_comic) \
			.flatMap(lambda i:i[1][1:]).distinct()
		level += 1 
		
		#gets rid of duplicate comic books from current iteration and 
		#duplicate characters from previous iteration
		data_comic=data_comic_next.subtractByKey(comic_list.map(lambda i:(i,1))).cache()
		data_char=data_comic.map(lambda i: (i[1],i[0])).cache()
		
		#stores the data set with removed characters for the next iteration
		#we don't want to use it for the next search since we've already removed
		#the characters we will be searching for.
		data_char_next=data_char.subtractByKey(char_list.map(lambda i:(i,1)))
		data_comic_next=data_char_next.map(lambda i: (i[1],i[0]))
		
		
		graph_level=char_list.map(lambda i:(i,level)).subtractByKey(graph).cache()
		graph_size.append(graph_level.count())
#		graph_level_arr.append(graph_level.collect())
		graph=graph+char_list.map(lambda i:(i,level)).cache() #union
		



#.flatMap(lambda i:chain.from_iterable(i[1])).distinct().collect()
	