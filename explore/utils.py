import os
import json
import random
import numpy as np

from collections import namedtuple


Dataset = namedtuple('Dataset', ["embedding", "labels", "runtimes", "censored", "instance_index", "label_index"])

def load_dataset(embed_file, label_file, fill_unknown = False):

    labeled_instances = set(_path_to_task_id(file_path) for file_path in _parse_labels(label_file).keys())
    
    instance_index = []
    embeddings = []

    with open(embed_file, "r") as stream:
        
        for line in stream:
            content = json.loads(line)
            path = content["path"]
            path_id = _path_to_task_id(path)

            if path_id not in labeled_instances: continue
            labeled_instances.discard(path_id)

            instance_index.append(path_id)
            embeddings.append(content["embedding"])

    labels = [None] * len(embeddings)
    runtime = [None] * len(embeddings)
    censored = [None] * len(embeddings)

    content  = _parse_labels(label_file)

    if fill_unknown:
        label_index = set.union(*[set(D.keys()) for D in content.values()])
    else:
        label_index = set.intersection(*[set(D.keys()) for D in content.values()])
    
    label_index = sorted(list(label_index))

    index = {k: i for i, k in enumerate(instance_index)}

    for file_path, V in content.items():
        task_id = _path_to_task_id(file_path)
        if task_id not in index: continue

        if fill_unknown:
            V = {k: V[k] if k in V else {"solve": False, "cputime": 900} for k in label_index}

        ix = index[task_id]
        labels[ix] = [1 if V[k]["solve"] else 0 for k in label_index]
        runtime[ix] = [V[k]["cputime"] for k in label_index]
        censored[ix] = [1 if V[k]["cputime"] >= 900 else 0 for k in label_index]
        
    unlabelled_instances = [instance_index[i] for i, label in enumerate(labels) if label is None]

    if len(unlabelled_instances) > 0:
        print(unlabelled_instances)
        print("Could not label %d instances" % len(unlabelled_instances))
        print("Reduce sets")

        mask = [instance not in unlabelled_instances for instance in instance_index]
        instance_index = [instance for i, instance in enumerate(instance_index) if mask[i]]
        embeddings     = [embedding for i, embedding in enumerate(embeddings) if mask[i]]
        labels         = [label for i, label in enumerate(labels) if mask[i]]
        runtime        = [rt for i, rt in enumerate(runtime) if mask[i]]
        censored       = [c for i, c in enumerate(censored) if mask[i]]


    return Dataset(np.array(embeddings), np.array(labels), np.array(runtime), np.array(censored), instance_index, label_index)


def _index_elements(obj, index):
    try:
        return obj[index]
    except Exception:
        index = set(index)
        return [o for i, o in enumerate(obj) if i in index]

def _path_to_task_id(path):
    if len(path.split("/")) == 2: return path
    base_name = os.path.basename(path)
    dirname = os.path.basename(os.path.dirname(path))
    base_name = base_name.replace(".c", "").replace(".i", "").replace(".yml", "")
    return f"{dirname}/{base_name}"


def _parse_labels(label_file):
    if label_file.endswith(".json"):
        with open(label_file, "r") as stream:
            content = json.load(stream)
    else:
        content = _parse_preprocess_file(label_file)
    return content

def stratified_split(dataset, ratio=0.3):
    
    # Groups with respect to folders
    groups = {}

    for i, path in enumerate(dataset.instance_index):
        g = os.path.basename(os.path.dirname(path))
        if g not in groups: groups[g] = []
        groups[g].append(i)

    # Sample indices according to groups
    train_indices, test_indices = [], []

    for G in groups.values():
        test_size = int(len(G) * ratio)
        test_ix   = random.sample(G, test_size)
        test_indices.extend(test_ix)
        train_ix  = [i for i in G if i not in test_ix]
        train_indices.extend(train_ix)

    # Split dataset
    test_dataset  = Dataset(*map(lambda x: _index_elements(x, test_indices), dataset[:-1]), dataset.label_index)
    train_dataset = Dataset(*map(lambda x: _index_elements(x, train_indices), dataset[:-1]), dataset.label_index)

    return train_dataset, test_dataset


def stratified_cv(dataset, num_splits = 3):
    
    # Groups with respect to folders
    groups = {}

    for i, path in enumerate(dataset.instance_index):
        g = os.path.basename(os.path.dirname(path))
        if g not in groups: groups[g] = []
        groups[g].append(i)

    # Sample indices according to groups
    partitions = [[] for _ in range(num_splits)]

    for G in groups.values():
        group_index = list(G)
        random.shuffle(group_index)

        for i, ix in enumerate(G):
            partitions[i % len(partitions)].append(ix)

    # Build cross val datasets
    for i, partition in enumerate(partitions):
        # Split dataset
        train_indices = [i for i in range(len(dataset.instance_index)) if i not in partition]
        test_dataset  = Dataset(*map(lambda x: _index_elements(x, partition), dataset[:-1]), dataset.label_index)
        train_dataset = Dataset(*map(lambda x: _index_elements(x, train_indices), dataset[:-1]), dataset.label_index)

        yield train_dataset, test_dataset

# Parsing labels --------------------------------------------------------

def _has_solved(entry):
    if "ERROR" in entry["status"]          : return False
    if "timeout" in entry["status"].lower(): return False

    return str(entry["verdict"]).lower() in entry["status"].lower()


def _parse_preprocess_file(label_file):
    content = {}

    with open(label_file, "r") as lines:
        for line in lines:
            entry = json.loads(line)
            task  = set(e["task_file"] for e in entry)
            assert len(task) == 1, str(task)
            task = next(iter(task))
            
            content[task] = {
                e["tool"]: {
                    "status": e["status"],
                    "cputime": e["cputime"],
                    "verdict": e["verdict"],
                    "solve": _has_solved(e)
                }
                for e in entry
            }

    return content
