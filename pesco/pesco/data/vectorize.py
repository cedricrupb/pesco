from collections import defaultdict

def traversal_to_counts(traversal):
    counter = defaultdict(int)
    
    for _, _, _, parent, *_ in traversal:
       counter[parent.type] += 1

    return dict(counter)


def counts_to_vector(counts, vocab):
   vector = [0] * len(vocab)

   for k, count in counts.items():
      if k in vocab:
         vector[vocab[k]] = count

   return vector