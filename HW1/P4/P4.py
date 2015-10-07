
# coding: utf-8

# In[298]:

#Author: Xingchi Dai
#import Spark
import findspark as fs;
fs.init();
import pyspark as py
from P4_bfs import *


# In[299]:

#create sparkConf and a new Spark
conf = py.SparkConf().setAppName("CS205HW1Q4")
sc = py.SparkContext();


# In[300]:

#create a new CSV file
rdd = sc.textFile("source.csv")


# In[301]:

#split the characters and their comic
splited_rdd = rdd.map(lambda x: (x.split('"')[3],x.split('"')[1]));
#then we group them by key
new_rdd = splited_rdd.groupByKey().map(lambda x : (x[0], list(x[1])));
new_rdd.cache();


# In[302]:

import copy

#This function processes the data we just got [book,[list of chars]]
#In order to get the neighbor, the function will pick one char from the list,
#the rest of chars become the neighbors of the one just picked from
#the function will return a result of multiple entries in a form of
# [char,[neighbors]]
def get_neighbor(x):
        Nlist = x[1];
        result = [];
        for x in range(len(Nlist)):
            ele = [];
            new_num = Nlist[x];
            ele.append(new_num);
            copy_list = copy.deepcopy(Nlist)
            copy_list.remove(new_num);
            ele.append(copy_list);
            result.append(ele);
        return result;
#create the neighbors for each characters
chars_rdd = new_rdd.flatMap(get_neighbor);
#we group the same char, and combine their neighbors
chars_rdd = chars_rdd.reduceByKey(lambda x,y: list(set(x + y)));
chars_rdd.cache();


print bfs(chars_rdd,"CAPTAIN AMERICA",sc)
print bfs(chars_rdd,"MISS THING/MARY",sc)
print bfs(chars_rdd,"ORWELL",sc)




