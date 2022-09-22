import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def wnn_score(dataset, alpha = 10, embedding = None, return_prediction = False):
    embedding    = embedding if embedding is not None else dataset.embedding
    similarities = -euclidean_distances(embedding)

    # Exclude own label from calculation
    self_mask    = np.eye(len(similarities))
    similarities = similarities * (1 - self_mask) - 1000 * self_mask 

    softmax_dist = alpha * similarities
    softmax_dist = np.exp(softmax_dist - np.max(softmax_dist, axis = 1, keepdims = True))
    softmax_dist /= softmax_dist.sum(axis = 1, keepdims = True)

    labels = dataset.labels
    wnn_labels = softmax_dist @ labels
    wnn_mask   = np.zeros(wnn_labels.shape)
    wnn_mask[np.arange(wnn_labels.shape[0]), wnn_labels.argmax(axis = 1)] = 1
    
    solvability = (wnn_mask * labels).sum(axis = 1).mean()
    print((1 - (wnn_mask * labels).sum(axis = 1)).sum())
    oracle = solvability / labels.max(axis = 1).mean()

    if return_prediction:
        return oracle, wnn_mask

    return oracle


def knn_score(dataset, k = 10, embedding = None, return_prediction = False):
    embedding    = embedding if embedding is not None else dataset.embedding
    similarities = cosine_similarity(embedding)

    # Exclude own label from calculation
    self_mask    = np.eye(len(similarities))
    similarities = similarities * (1 - self_mask) - 1000 * self_mask 

    top_similarities = similarities.argsort(axis = 1)[:, ::-1]
   
    top_mask = np.zeros(similarities.shape)
    for i, row in enumerate(top_similarities):
        for j in row[:k]: 
            top_mask[i, j] = 1

    softmax_dist = top_mask / top_mask.sum(axis = 1, keepdims = True)

    labels = dataset.labels
    wnn_labels = softmax_dist @ labels
    wnn_mask   = np.zeros(wnn_labels.shape)
    wnn_mask[np.arange(wnn_labels.shape[0]), wnn_labels.argmax(axis = 1)] = 1
    
    solvability = (wnn_mask * labels).sum(axis = 1).mean()
    oracle = solvability / labels.max(axis = 1).mean()

    if return_prediction:
        return oracle, wnn_mask

    return oracle

# Better scores --------------------------------

def wnn_ind_scores(embedding, labels, alpha = 1, k = 1, score_labels = None):
    similarities = -euclidean_distances(embedding)

    # Exclude own label from calculation
    self_mask    = np.eye(len(similarities))
    similarities = similarities * (1 - self_mask) - 1000 * self_mask 

    softmax_dist = alpha * similarities
    softmax_dist = np.exp(softmax_dist - np.max(softmax_dist, axis = 1, keepdims = True))
    softmax_dist /= softmax_dist.sum(axis = 1, keepdims = True)

    wnn_labels = softmax_dist @ labels

    wnn_arg_index = wnn_labels.argsort(axis = 1)[:, ::-1]
    wnn_mask   = np.zeros(wnn_labels.shape)

    for i in range(k):
        wnn_mask[np.arange(wnn_labels.shape[0]), wnn_arg_index[:, i]] = 1

    if score_labels is None: score_labels = labels
    
    solvability = (wnn_mask * score_labels).max(axis = 1)
    oracle = solvability.mean() #/ labels.max(axis = 1).mean()
    predictions = (wnn_mask * score_labels).sum(axis = 1)

    return oracle, predictions