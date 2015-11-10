# Initialize SC context
import numpy as np
from P5_bfs import *
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf

# Your code here
if __name__ == '__main__':
  #Setup
  conf = SparkConf().setAppName("wikipedia_graph")
  sc = SparkContext(conf=conf)
  # For local use
  #links = sc.textFile('links-simple-sorted.txt')
  #page_names = sc.textFile('titles-sorted.txt')
  # For AWS use
  links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
  page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')

  # links data stored into RDD (node, neighbors)
  rdd_links = links.map(lambda data: (data.split(":")[0],data.split(":")[1].lstrip()))
  rdd_links = rdd_links.map(lambda (node, neighbors): (node, neighbors.split()))

  # Pages data stored into RDD (page, index)
  rdd_pages = page_names.zipWithIndex().map(lambda (page, index): (page, str(index+1)))
  # Pages data stored into RDD (index, page)
  rdd_index_page = rdd_pages.map(lambda (page,index): (index,page))
  
  # Lookup corresponding nodes for "Kevin Bacon" and "Harvard University"
  KB = "Kevin_Bacon"
  HU = "Harvard_University"
  KB_node = rdd_pages.map(lambda x : x).lookup(KB)[0]
  HU_node = rdd_pages.map(lambda x : x).lookup(HU)[0]
 
  # "Kevin_Bacon" -> "Harvard University"
  rdd_KB_HU, level_KB_HU  = BFS_with_short_path(rdd_links, KB_node, HU_node)
  short_path_KB_HU_labels, short_path_KB_HU_pages = Generate_Short_Path(rdd_KB_HU, rdd_index_page, KB_node, HU_node)

  # "Harvard University" -> "Kevin Bacon"
  rdd_HU_KB, level_HU_KB = BFS_with_short_path(rdd_links, HU_node, KB_node)
  short_path_HU_KB_labels, short_path_HU_KB_pages = Generate_Short_Path(rdd_HU_KB, rdd_index_page, HU_node, KB_node)

  print "Kevin_Bacon -> Harvard University"
  print "diameter : %s " % level_KB_HU
  print "short path using labels : %s" % short_path_KB_HU_labels
  print "short path using pages : %s" % short_path_KB_HU_pages
  print "Harvard University -> Kevin Bacon"
  print "diameter : %s " % level_HU_KB
  print "short path using labels : %s" % short_path_HU_KB_labels
  print "short path using pages : %s" % short_path_HU_KB_pages
