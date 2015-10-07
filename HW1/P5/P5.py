
# coding: utf-8

# In[1]:

#Author: Xingchi Dai
#import spark files
import findspark as fs;
fs.init();
import pyspark as py


# In[2]:

#create sparkConf and a new Spark
conf = py.SparkConf().setAppName("CS205HW1Q4")
sc = py.SparkContext();


# In[41]:

#create a new CSV file
#use the local files to run
titles_rdd = sc.textFile("titles-sorted.txt");
links_rdd = sc.textFile("links-simple-sorted.txt");
#this rdd is to use to check the id of one title
titles_rdd = titles_rdd.zipWithIndex().map(lambda x: (x[0], x[1] + 1))
#this rdd is to check the title of one idd
titles_rdd_id = titles_rdd.map(lambda x: (x[1],x[0]));
titles_rdd.cache();
#split the link to form a rdd graph
links_rdd = links_rdd.map(lambda x: tuple(x.split(":"))).map(lambda x:(int(x[0]),x[1].split()));


# In[61]:

#now find the shortest path
import math
import copy

#this function initalize every node, set
#the starting node as distance 0, others
#are infinity
def init_nodes(x,start_node):
    if(x[0] == start_node):
        return (x[0],(0,0,x[1]));
    return (x[0],(float('inf'),0,x[1]));

#this function is to update all nodes
#this function is the center of the algotithm
#the function checks the current level, and 
#traverse its neighbors, which is the nexe lever,
#all nodes are stored in the form of
#[name,[distance,parent_node,[children]]
def new_entries(x,level):
    result = [];
    #still inf,already constructed as in the init
    result.append(x);
    par = copy.copy(x[0]);
    #find the parent level
    if (x[1][0] == level - 1):
        for entry in x[1][2]:
            result.append((int(entry),(level,par,[])));#we need to store the parent info 
    return result;

#parent should be only shown once in the list
#and this should store the shortest path
#the reconstruct function will find the shortest 
#path, and stor its parent node
def reconstruct(x,y):
    level = min(x[0], y[0]);
    parent = x[1] if (level == x[0]) else y[1];
    new_neighbor = set(x[2] + y[2]);
    return (level,parent,list(new_neighbor));

#in this function, we need to write the path into the file
def write_into_the_file(source,path,file_name):
    
    final_path = "The path is: "
    for i in result:
        final_path = final_path + "".join(i) + "--->"
    #open the file
    f = open(file_name,'w');
    f.write(final_path);
    f.write('\n');
    f.close();
    #Also print it in the terminal
    print final_path;
    
#should be a directed bfs
def bfs(graph,start_point,end_point,sc):
    #we need to transfer the start name and end_name to indexes first
    start_name =  int(titles_rdd.lookup(start_point)[0]);
    end_name = int(titles_rdd.lookup(end_point)[0]);
    number_of_nodes = 0;
    graph = graph.map(lambda x: init_nodes(x,start_name));
    accum = sc.accumulator(0);# shall we use accumulator in this problem?
    #move to the next level
    for level in range(10): 
        #used rdd.filter to optimize the amount of data transffered
        graph = graph.flatMap(lambda x: new_entries(x,level));
        graph = graph.reduceByKey(lambda x,y: reconstruct(x,y));
        if not(math.isinf(graph.lookup(end_name)[0][0])):
              break;
#we need to print out the shortest-path in this question
#find the end point
    short_path = [];
    parent = end_name;
    while(True):
        if(parent == start_name):
            break;
        else:
            short_path.append(parent); 
            parent = graph.lookup(parent)[0][1];
    short_path.append(start_name);
    
    write_into_the_file(titles_rdd_id,short_path,"P5.txt");
    
    return "Success";

# Run the program
bfs(links_rdd,"Kevin_Bacon","Harvard_University",sc);
bfs(links_rdd,"Harvard_University","Keven_Bacon",sc);


# In[ ]:



