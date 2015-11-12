from  pyspark import SparkContext
sc = SparkContext()
sc.setLogLevel("ERROR")

#page_names = sc.textFile('titles-sorted.txt')

page_names = page_names.zipWithIndex().mapValues(lambda V: V+1).map(lambda (K,V): (V,K))

#print page_names.lookup(2578703)
#print page_names.lookup(4625677)
#print page_names.lookup(1124925)
#print page_names.lookup(3229511)
#print page_names.lookup(5114592)

print page_names.lookup(1786605)
print page_names.lookup(5438030)


