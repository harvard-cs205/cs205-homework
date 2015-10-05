import random
total_nodes = 100000
max_out = 50
start_nodes = range(1, total_nodes)


for node in start_nodes:
	num_picked = random.randint(1, max_out)
	items = map(str, random.sample(start_nodes, num_picked))
	print node, ":", ' '.join(items)
