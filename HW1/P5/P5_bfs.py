"""
Written by Jaemin Cheun
Harvard CS205
Assignment 1
October 6, 2015
"""
import numpy as np
import findspark
findspark.init()

from pyspark import SparkContext

def link_split(line):
	line = line.split(": ")
	return(int(line[0]), map(int, line[1].split(" ")))

# initiaize Spark
sc = SparkContext("local", appName="P5")
sc.setLogLevel("ERROR")

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')

index_pages = page_names.zipWithIndex().map(lambda x: (x[1] + 1, x[0])).cache()

graph = links.map(link_split).cache()

# find index for Kevin Bacon and Harvard University
index_Kevin = index_pages.filter(lambda (k,v): v == "Kevin_Bacon").collect()[0][0]
index_Harvard = index_pages.filter(lambda (k,v): v == "Harvard_University").collect()[0][0]
