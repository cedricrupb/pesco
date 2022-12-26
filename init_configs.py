#!/usr/bin/env python3
import os
import sys
import json

from glob import glob

lib_dir = os.path.join(os.path.dirname(__file__), "lib", "python")
sys.path.insert(0, lib_dir)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pesco"))

import joblib
from pesco.cpachecker import generate_cpa_config

for selector_path in glob(os.path.join("resource", "*.jbl")):
    print("Load selector:", selector_path)
    selector = joblib.load(selector_path)

    print("Generate configs...")
    try:
        for tool in selector._tools:
            tool = selector._prediction_to_label(tool)
            print("Generate config: ", tool)
            generate_cpa_config(tool)
    except KeyError:
        print("Selector did not need generation:", selector_path)

mapping_path = os.path.join("resource", "mapping.json")
if os.path.exists(mapping_path):
    print("Generate mapping configs")
    with open(mapping_path, "r") as i:
        mapping = json.load(i)

    for tool in mapping.values():
        if tool == "passthrough": continue
        print("Generate config: ", tool)
        generate_cpa_config(tool)
