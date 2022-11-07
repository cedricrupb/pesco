import unittest
import os
import sys

lib_dir = os.path.join(os.path.dirname(__file__), "..", "lib", "python")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pesco")) # Add PeSCo also
sys.path.insert(0, lib_dir)

if "PYTHONPATH" not in os.environ:
    os.environ["PYTHONPATH"] = ""
os.environ["PYTHONPATH"] += os.pathsep + str(
    lib_dir
)  # necessary so subprocesses also use libraries

import json
import numpy as np

from pesco.data import clang_to_traversal, traversal_to_counts, counts_to_vector


class TestASTCounter(unittest.TestCase):

    def setUp(self):
        # Use clang for feature extraction
        try:
            clang_executable = os.path.join("lib", "clang", "clang")
        except ValueError:
            clang_executable = None
        self.clang_executable = clang_executable

    def parse_code_to_counts(self, program_path):
        traversal = clang_to_traversal(program_path, 2, clang_executable = self.clang_executable)
        return traversal_to_counts(traversal)
    
    def test_ast_count_1(self):
        
        with open("./tests/programs/array-2.json", "r") as i:
            target_counts = json.load(i)["count"]
        
        counts = self.parse_code_to_counts("./tests/programs/array-2.c")

        self.assertEqual(set(counts.keys()), set(target_counts.keys()))

        for key, count in counts.items():
            with self.subTest(key = key):
                self.assertEqual(count, target_counts[key])

    def test_ast_count_2(self):
        
        with open("./tests/programs/data_structures_set_multi_proc_ground-1.json", "r") as i:
            target_counts = json.load(i)["count"]
        
        counts = self.parse_code_to_counts("./tests/programs/data_structures_set_multi_proc_ground-1.i")

        self.assertEqual(set(counts.keys()), set(target_counts.keys()))

        for key, count in counts.items():
            with self.subTest(key = key):
                self.assertEqual(count, target_counts[key])

    def test_ast_count_3(self):
        
        with open("./tests/programs/Problem03_label00.json", "r") as i:
            target_counts = json.load(i)["count"]
        
        counts = self.parse_code_to_counts("./tests/programs/Problem03_label00.c")

        self.assertEqual(set(counts.keys()), set(target_counts.keys()))

        for key, count in counts.items():
            with self.subTest(key = key):
                self.assertEqual(count, target_counts[key])

# Test embedding -----------------------------------------------------------


class TestEmbedding(unittest.TestCase):

    def setUp(self):
        # Use clang for feature extraction
        try:
            clang_executable = os.path.join("lib", "clang", "clang")
        except ValueError:
            clang_executable = None
        self.clang_executable = clang_executable

    def extract_features_from_program(self, program_path):
    
        # Load vocabulary
        vocab_path = os.path.join("tests", "models", "vocab.txt")
        vocab = {}
        with open(vocab_path, "r") as lines:
            for i, line in enumerate(lines):
                vocab[line.strip()] = i

        # Parse file
        traversal = clang_to_traversal(program_path, 2, clang_executable = self.clang_executable)
        counts    = traversal_to_counts(traversal)
        vector    = counts_to_vector(counts, vocab)

        return np.array(vector)

  
    def test_embedding_1(self):
        
        with open("./tests/programs/array-2.json", "r") as i:
            target_embedding = np.array(json.load(i)["embedding"])
        
        embedding = self.extract_features_from_program("./tests/programs/array-2.c")

        self.assertEqual(embedding.shape[0], target_embedding.shape[0])

        for i in range(embedding.shape[0]):
            with self.subTest(i = i):
                self.assertEqual(embedding[i], target_embedding[i])

    def test_embedding_2(self):
        
        with open("./tests/programs/data_structures_set_multi_proc_ground-1.json", "r") as i:
            target_embedding = np.array(json.load(i)["embedding"])
        
        embedding = self.extract_features_from_program("./tests/programs/data_structures_set_multi_proc_ground-1.i")

        self.assertEqual(embedding.shape[0], target_embedding.shape[0])

        for i in range(embedding.shape[0]):
            with self.subTest(i = i):
                self.assertEqual(embedding[i], target_embedding[i])

    def test_embedding_3(self):
        
        with open("./tests/programs/Problem03_label00.json", "r") as i:
            target_embedding = np.array(json.load(i)["embedding"])
        
        embedding = self.extract_features_from_program("./tests/programs/Problem03_label00.c")

        self.assertEqual(embedding.shape[0], target_embedding.shape[0])

        for i in range(embedding.shape[0]):
            with self.subTest(i = i):
                self.assertEqual(embedding[i], target_embedding[i])




if __name__ == "__main__":
    unittest.main()