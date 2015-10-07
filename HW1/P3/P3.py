#import the pySpark
import pyspark as py


#This function helps to create a new key/pair of RDD
#return the final result
def finalResult(x):
	newList = list(x[1]);
	return (x[0],len(newList),newList);


#initialize the spark
conf = py.SparkConf().setAppName("CS205HW1P3")
sc = py.SparkContext();
#create a new RDD
wlist = sc.textFile('EOWL_words.txt');
#return a key-value pair
pairs = wlist.map(lambda x: (''.join(sorted(x)),x));
#use groupbyKey to group the same word and sort the largest number of valid anagrams
result = pairs.groupByKey().map(finalResult).takeOrdered(1,key = lambda x: -x[1]);
#print the result
out_file = open('P3.txt','w');
out_file.write(str(result));
out_file.close();


