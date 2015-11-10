import numpy as np
from P5_bfsCC import *


def P5_connected_component(link_edges,sc):
	component_sizes = []
	while not link_edges.isEmpty():
		
		#Select arbitrary node to initialize search
		page = link_edges.take(1)[0][0] 
		
		compOFpage, COUNTcompOFpage = P5_bfsCC(link_edges, page)
		compRDD = sc.parallelize(compOFpage.keys()).map(lambda x: (x,0))
		
		#Remove component from graph
		link_edges = link_edges.subtractByKey(compRDD).cache() 
		
		#Record component size
		component_sizes.append(COUNTcompOFpage)
		
	return component_sizes #Return array of component sizes