# Start the cluster
import seaborn as sns
sns.set_context('poster', font_scale=1.25)
import findspark as fs
fs.init()
import pyspark as ps
import multiprocessing as mp

### Setup ###
config = ps.SparkConf()
# Assumes setup is in spark.conf
config = config.setAppName('marvel_solver')
sc = ps.SparkContext(conf=config)

# Read in the data

marvel_raw_data = sc.textFile('marvel_data.csv')

def get_issue_character(raw_line):
    dat = raw_line.split('"')
    character_name = dat[1]
    issue = dat[3]
    return (issue, character_name)

issue_character_rdd = marvel_raw_data.map(get_issue_character)
# Put everyone in a group if they are in the same comic and get nodes from that!
issue_groups_rdd = issue_character_rdd.groupByKey()

def get_links_from_groups(x):
    list_of_characters = list(x[1])
    links = []
    for cur_character in list_of_characters:
        for other_character in list_of_characters:
            if cur_character != other_character:
                # Ensure that links are symmetric
                links.append((cur_character, other_character))
                links.append((other_character, cur_character))
    return tuple(links)

all_links_rdd = issue_groups_rdd.flatMap(get_links_from_groups)
character_and_links_rdd = all_links_rdd.groupByKey()

# There are a bunch of duplicates; we want to get rid of them
def cleanup_links(x):
    linked_to = list(x[1])
    unique_links = tuple(set(linked_to))
    return (x[0], unique_links)

network_rdd = character_and_links_rdd.map(cleanup_links).cache() # Completed network representation
# Our network has the form (Name, [All individuals that the node links to]). Note that
# in this representation (A, [B]) does not imply (B, [A]), i.e. this is an innately directed
# representation, but we initialize the network in such a way that the links are bi-directional.

# We now do the breadth-first search, BFS
from HW1.network_commands import BFS

#### Captain America ####
searcher = BFS(sc, 'CAPTAIN AMERICA', network_rdd)
searcher.run_until_converged() # I could run 10 iterations, but the solution converges faster than that
result = searcher.distance_rdd
america_connections= searcher.distance_rdd.count() - 1

#### Miss Thing/Mary #####
searcher = BFS(sc, 'MISS THING/MARY', network_rdd)
searcher.run_until_converged()
mary_connections= searcher.distance_rdd.count() - 1

#### Orwell ####
searcher = BFS(sc, 'ORWELL', network_rdd)
searcher.run_until_converged()
orwell_connections= searcher.distance_rdd.count() - 1

#Print everything at the end
print 'Captain america is connected to' , america_connections, 'other characters.'
print 'MISS THING/MARY is connected to' , mary_connections, 'other characters.'
print 'ORWELL is connected to' , orwell_connections, 'other characters.'