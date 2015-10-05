
# coding: utf-8

# In[1]:

#import findspark
#import os
#findspark.init('/home/chongmo/spark') # you need that before import pyspark.
#import pyspark
#from pyspark import SparkContext
sc=pyspark.SparkContext()
sc.setLogLevel("WARN")

# In[3]:

link_RDD=sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)


# In[6]:

title_RDD=sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)


# In[4]:

neighbors_RDD=link_RDD.map(lambda x: x.split(': ')).mapValues(lambda x: x.split(' ')).partitionBy(256).cache()


# In[7]:

title_RDD=title_RDD.zipWithIndex().map(lambda x: (x[1]+1, x[0])).sortByKey().cache()


# In[11]:

Kevin_Bacon = str(title_RDD.filter(lambda x: x[1] == 'Kevin_Bacon').first()[0])


# In[12]:

Harvard_University = str(title_RDD.filter(lambda x: x[1] == 'Harvard_University').first()[0])


# In[ ]:
def BFS(G, T, start, goal): #G:graph RDD, (char1, [char2, char3, ...])    
    queue = sc.parallelize([start])
    prev ={}# prev[nd] = previous node in the path before nd
    while not queue.isEmpty():
        q=set(queue.collect())
        neighbors =G.filter(lambda x: x[0] in q).flatMapValues(lambda x:x).filter(lambda x: not(x[1] in set(prev.keys()))).map(lambda x: (x[1],x[0])).reduceByKey(lambda x,y: x).distinct()
        prev.update(dict(neighbors.collect())) 
        #print prev
        if goal in set(neighbors.keys().collect()):
            path = [T.filter(lambda x: x[0]==int(goal)).first()[1]]
            cur = goal
            while cur!=start: 
                cur = prev[cur]
                path.append(T.filter(lambda x: x[0]==int(cur)).first()[1])
            path = path[::-1]
            return path
        queue = neighbors.keys()
        #print queue.collect()
    return 'no'

path=BFS(neighbors_RDD, title_RDD, Kevin_Bacon, Harvard_University)
#path=BFS(neighbors_RDD, title_RDD, Harvard_University, Kevin_Bacon)

print path
