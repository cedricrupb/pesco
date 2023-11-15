import os
import sys

lib_dir = os.path.join(os.path.dirname(__file__), "..", "lib", "python")

sys.dont_write_bytecode = True  # prevent creation of .pyc files
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pesco")) # Add PeSCo also
sys.path.insert(0, lib_dir)

if "PYTHONPATH" not in os.environ:
    os.environ["PYTHONPATH"] = ""
os.environ["PYTHONPATH"] += os.pathsep + str(
    lib_dir
)  # necessary so subprocesses also use libraries


import os
import json

import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_folder")
    args = parser.parse_args()

    with open(args.input_file, "r") as input:
        content = [json.loads(i) for i in input]

    vocab_path = os.path.join(args.output_folder, "vocab.txt")

    if os.path.exists(vocab_path):
        print("Load vocabulary")
        with open(vocab_path, "r") as inp:
            vocab = {l.rstrip(): i for i, l in enumerate(inp)}
    else:
        vocab = set.union(*[set(c["counts"]) for c in content])
        vocab  = {k: i for i, k in enumerate(sorted(vocab))}
        with open(vocab_path, "w") as o:
            v = {v: k for k, v in vocab.items()}
            for i in range(len(vocab)):
                o.write(v[i] + "\n")

    v = {v: k for k, v in vocab.items()}
    rvocab = [v[i] for i in range(len(vocab))]

    with open(os.path.join(args.output_folder, "embedding.jsonl"), "w") as o:
        for entry in content:
            counts = entry["counts"]
            o.write(
                json.dumps({
                    "path": entry["file_path"],
                    "embedding": [counts.get(k, 0) for k in rvocab]
                }) + "\n" 
            )


if __name__ == '__main__':
    main()