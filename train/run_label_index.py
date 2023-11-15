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


import argparse
import json

def read_jsonl(file_path):
    with open(file_path, "r") as i:
        for line in i:
            yield json.loads(line)
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label_file")
    parser.add_argument("output")

    parser.add_argument("--tool", type = str, default = "")
    parser.add_argument("--status", type = str, default = "")
    parser.add_argument("--negate", action = "store_true")

    args = parser.parse_args()

    index = set()

    for entry in read_jsonl(args.label_file):
        for result in entry:
            if args.tool   not in result["tool"]: continue

            next = not args.negate
            for status in args.status.split(","):
                if args.negate and result["status"].startswith(status): next = True
                if not args.negate and result["status"].startswith(status): next = False
            
            if next: continue
            
            index.add(result["task_file"])
    
    print("Index:", len(index))
    
    with open(args.output, "w") as o:
        for file_path in sorted(index):
            o.write(file_path + "\n")

if __name__ == "__main__":
    main()