import numpy as np
from P5_bfs import *


def P5_connected_component(link_edges,sc):
	component_sizes = []
	while not link_edges.isEmpty():
		
		#Select arbitrary node to initialize search
		page = link_edges.take(1)[0][0] 
		
		compOFpage, COUNTcompOFpage = P5_bfs(link_edges, page, sc)
		
		#Remove component from graph
		link_edges = link_edges.subtractByKey(compOFpage).partitionBy(32).cache() 
		
		#Record component size
		component_sizes.append(COUNTcompOFpage)
		
	return component_sizes #Return array of component sizes