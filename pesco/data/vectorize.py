from collections import defaultdict

def traversal_to_counts(traversal):
    counter = defaultdict(int)
    
    for _, _, _, parent, *_ in traversal:
       counter[parent.type] += 1

    return dict(counter)