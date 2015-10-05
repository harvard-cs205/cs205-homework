from collections import defaultdict

d = {'1': 6, '2': 1, '3': 1, '4': 9, '5': 9, '6': 1}

v = defaultdict(list)

for key, value in sorted(d.iteritems()):
    v[value].append(key)

print v