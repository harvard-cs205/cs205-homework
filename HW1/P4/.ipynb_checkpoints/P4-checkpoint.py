{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "import findspark\n",
    "findspark.init()\n",
    "import pyspark\n",
    "import json\n",
    "sc = pyspark.SparkContext(appName=\"\")\n",
    "# from P4_bfs import bfs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "## Import the character-comic pair.\n",
    "source = sc.textFile('source.csv')\n",
    "\n",
    "## For each comic (key), group the comic's characters into a list, then remove the comic key.\n",
    "def split_raw(x):\n",
    "    x1 = x.strip('\"').split('\",\"')\n",
    "    if len(x1)==2:\n",
    "        return (x1[1], x1[0])\n",
    "source1 = source.map(split_raw).groupByKey().map(lambda x: list(x[1]))\n",
    "## We are now left with lists of characters in each comic.\n",
    "\n",
    "## In each list of characters, use each character as the key to create a list of character's neighbors.\n",
    "## Then combine different neighbor lists for each character.\n",
    "def comic_to_neighbor(vlist):\n",
    "    neighbor = []\n",
    "    for i in range(len(vlist)):\n",
    "        neighbor.append((vlist[i], vlist[:i] + vlist[i+1:]))\n",
    "    return neighbor\n",
    "source2 = source1.flatMap(comic_to_neighbor).reduceByKey(lambda x,y: list(set(x+y))).map(lambda x: (x[0], (1000, x[1]))).cache()\n",
    "## Now we have (k, v) for k=character and v=neighbors.\n",
    "\n",
    "result = bfs(source2, 'Captain America', 10)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def bfs_core(inputRDD, step, visit_count):\n",
    "    def find_neighbor((node, (dist, neighbors))):\n",
    "        next_list = [(node, (dist, neighbors))]\n",
    "        if dist == step:\n",
    "            for neighbor in neighbors:\n",
    "                next_list.append((neighbor, (dist+1, [])))\n",
    "        return next_list\n",
    "    \n",
    "    outputRDD = inputRDD.flatMap(find_neighbor).reduceByKey(lambda (x1,y1), (x2,y2): (min(x1,x2), y1+y2))\n",
    "    return outputRDD\n",
    "\n",
    "\n",
    "def bfs(sourceRDD, root, step_limit = False):\n",
    "    ## Set the root\n",
    "    def set_root((node, (dist, neighbors))):\n",
    "        if (node == root):\n",
    "            return (node, (0, neighbors))\n",
    "        else:\n",
    "            return ((node, (dist, neighbors)))\n",
    "    nodeRDD = sourceRDD.map((set_root))\n",
    "    new_count = 1\n",
    "    visit_count = sc.accumulator(1)\n",
    "    \n",
    "    if step_limit != False:\n",
    "        for step in range(step_limit):\n",
    "            nodeRDD = bfs_core(nodeRDD, step, visit_count)\n",
    "            new_count = nodeRDD.filter(lambda (k, v): v[0] == step+1).count()\n",
    "            visit_count += new_count        \n",
    "            print \"Step: \", (step+1), \"; New Nodes: \", new_count, \"; Total Nodes: \", visit_count, \".\"\n",
    "        return (step+1, visit_count, nodeRDD)\n",
    "    else:\n",
    "        step = 0\n",
    "        while new_count > 0:\n",
    "            nodeRDD = bfs_core(nodeRDD, step, visit_count)\n",
    "            new_count = nodeRDD.filter(lambda (k, v): v[0] == step+1).count()\n",
    "            visit_count += new_count\n",
    "            print \"Step: \", (step+1), \"; New Nodes: \", new_count, \"; Total Nodes: \", visit_count, \".\"\n",
    "            if new_count == 0:\n",
    "                return (step+1, visit_count, nodeRDD)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
