from pyspark import SparkContext
from P4_bfs import BFS

def split_up_input(line):
    """ Takes in a string, assumed to be of the form '"character", "issue"'
    and returns a tuple of (character, issue)"""

    split_line = line.split('"')
    character = split_line[1]
    issue = split_line[3]

    return (character, issue)

def get_all_individual_pairs(issue_list_tup):
    """ Takes a list of characters, all of which appear are assumed to appear in 
    the same issue, and returns all combinations of (char1, [char2]). This is
    sort of like a really poorly thought out adjacency matrix. For use in conjunction with flatMap
    and to be post-processed into an adjacency list representation.
    """

    list_of_chars = issue_list_tup[1]

    edges = []
    for c1 in list_of_chars:
        for c2 in list_of_chars:
            if (c1 != c2):              # avoid self-links 
                edges.append((c1, [c2]))
                edges.append((c2, [c1]))

    return tuple(edges)                 # flatMap expects a tuple

if __name__ == "__main__":

    # First get a SparkContext object
    sc = SparkContext('local', 'BFS')

    # Now load in the data
    data = sc.textFile('source.csv')

    # Get rid of the stupid quotes
    chars_and_issues = data.map(split_up_input)

    # Flip it around and wrap in a list
    issues_and_chars = chars_and_issues.map(lambda (x, y): (y, [x]))

    # Now we have the issues as keys - all things with the same issue should be connected
    # Thus reduceByKey gets us (almost) to an adjacency list representation
    issues_and_chars = issues_and_chars.reduceByKey(lambda x, y: x + y)

    # Now we access every single edge that appears in the graph
    all_edges = issues_and_chars.flatMap(get_all_individual_pairs)

    # Now each key is a character, and it appears many times -
    # once for each edge. Consolidate all of these into an adjacency
    # list representation of the graph.
    marvel_graph = all_edges.reduceByKey(lambda x, y: x + y)

    # Clean up any duplicates that may appear
    marvel_graph = marvel_graph.map(lambda (x, y): (x, list(set(y))))
    
    # Cache our graph to speed up the searches
    marvel_graph.cache()

    # Now run the first BFS
    ca_graph = BFS(marvel_graph,'CAPTAIN AMERICA', 10)

    # Filter out all nodes who never got their distances updated
    num_touched_ca = ca_graph.filter(lambda (x, y): True if y[0] < 10**8 else False).count()

    # Now run the second BFS
    mtm_graph = BFS(marvel_graph,'MISS THING/MARY', 10)

    # Filter out all nodes who never got their distances updated
    num_touched_mtm = mtm_graph.filter(lambda (x, y): True if y[0] < 10**8 else False).count()

    # Now run the last BFS
    o_graph = BFS(marvel_graph,'ORWELL', 10)

    # Filter out all graphs who never got their distances updated
    num_touched_o = o_graph.filter(lambda (x, y): True if y[0] < 10**8 else False).count()

    print 'Num touched for orwell:', num_touched_o
    print 'Num touched for captain america:', num_touched_ca
    print 'Num touched for miss thing:', num_touched_mtm
