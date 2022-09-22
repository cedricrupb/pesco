
import os
import argparse

from glob import glob
from tqdm import tqdm

import json
import multiprocessing as mp

from pesco.data import clang_to_traversal, traversal_to_counts




# Map multiprocessing ----------------------------------------------------------------

def pmap(map_fn, data):

    cpu_count = mp.cpu_count()

    if cpu_count <= 4: # Too few CPUs for multiprocessing
        for output in map(map_fn, data):
            yield output

    with mp.Pool(processes = cpu_count) as pool:
        for output in pool.imap_unordered(map_fn, data, chunksize = 4 * cpu_count):
            yield output

# Main function ----------------------------------------------------------------

def vectorize(file_path):
    counts = traversal_to_counts(clang_to_traversal(file_path, 2))
    return {"file_path": file_path, "counts": counts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_file")

    parser.add_argument("--parallel", action="store_true", default=False)

    args = parser.parse_args()

    files = glob(os.path.join(args.input_dir, "**", "*.c"), recursive = True)
    files +=  glob(os.path.join(args.input_dir, "**", "*.i"), recursive = True)

    files = [f for f in files if " " not in f]
    files = sorted(files, key = lambda x: os.stat(x).st_size)

    map_fn = pmap if args.parallel else map

    with open(args.output_file, "w") as o:
        for output in tqdm(map_fn(vectorize, files), total = len(files)):
            o.write(json.dumps(output) + "\n")


if __name__ == '__main__':
    main()