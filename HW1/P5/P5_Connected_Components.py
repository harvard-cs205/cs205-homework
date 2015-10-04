from pyspark import SparkContext
import numpy

def visit(node):
	

sc=SparkContext()
txt = sc.textFile('links-simple-sorted.txt')
graph = txt.map(lambda s:(s.split(' ')[0][:-1], tuple(s.split(' ')[1:]),False  )  )
while len(node=graph.filter(lambda x:x[2]==False).collect())!=0:
	visit(node)
