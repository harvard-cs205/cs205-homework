import pyspark
from pyspark import SparkContext, SparkConf
sc = SparkContext()



wlist = sc.textFile("EOWL_words.txt")
#wlist.collect()

def alphabetical(my_string):
    inp = my_string
    chars = list(my_string)
    sorted_chars = sorted(chars)
    sorted_string = ''.join(sorted_chars)
    tpl = (sorted_string, [inp])
    return tpl
sorted_list = wlist.map(alphabetical)
#sorted_list.collect()

reduced_list = sorted_list.reduceByKey(lambda a,b: a+b)
#reduced_list.collect()

final_list = reduced_list.map(lambda (a,b): (a, len(b), b))
#final_list.collect()

maxRRD = final_list.max(key = lambda c: c[1])
#maxRRD

maxRRD_count = maxRRD[1]
# maxRRD_count
#final_list.filter(lambda d: d[1]==maxRRD_count)

maxRRD_wordlist = maxRRD[2]
#maxRRD_wordlist



myfile = open('P3.txt', 'w')
# myfile.write(str(maxRRD_count) + '\n')
# myfile.write(str(maxRRD_wordlist) + '\n')
myfile.write(str(maxRRD))
myfile.close()