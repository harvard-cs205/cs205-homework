from pyspark import SparkContext

def sortletter(word):
    return ''.join(sorted(word)), (1, [word])

def countunion(v1, v2):
    return v1[0]+v2[0], v1[1]+v2[1]

def flatten(r):
    return r[0], r[1][0], r[1][1]

if __name__ == '__main__':
    sc = SparkContext("local", appName="Spark1")
    sc.setLogLevel('WARN')
    
    #wlist = sc.textFile('https://s3.amazonaws.com/Harvard-CS205/wordlist/EOWL_words.txt', use_unicode=False)
    wlist = sc.textFile('EOWL_words.txt', 32, use_unicode=False)
    
    sortedletter = wlist.map(sortletter) #generate (key, value) pairs
    
    result = sortedletter.reduceByKey(countunion) #reduce word count and word lists
    
    flatresult = result.map(flatten) #tail the format
    
    collectresult = flatresult.collect() #this is the required RDD
    
    maxentry = max(collectresult, key=lambda x:x[1])
    print maxentry   
    



