import os
import shutil
from glob import glob

from .utils import execute, resolve_path


def find_test_case():
    if not os.path.exists("test-suite"): return None

    xml_files = [f for f in glob("test-suite/*.xml") if not f.endswith("metadata.xml")]
    if len(xml_files) > 1:
        print("Multiple test cases! Which should I pick?")
        return None

    return xml_files[0] if len(xml_files) > 0 else None



class Tester:

    def __init__(self, tool_name, version = None, witness = False):
        self.tool_name = tool_name
        self.version   = version
        self.witness   = witness

    def _build_cmd(self, executable, program_path, data_model = "LP64", cputime = None, memory = None, property_file = None):
        cmd  = ["python3", executable, self.tool_name, program_path]
        cmd += ["--data_model", data_model]

        if self.version is not None:
            cmd += ["--version", self.version]

        if cputime is not None:
            cmd += ["--cputime", str(cputime)]

        if memory is not None:
            cmd += ["--memory", str(memory)]

        if property_file is None:
            cmd += ["--property_file", property_file]
        
        cmd += ["--tool_directory", resolve_path("lib"), "--enforce_limits"]

        return cmd

    def _gen_witness(self, program_path, test_case, data_model = "LP64", property_file = None):
        executable = resolve_path("lib", "python", "test2witness", "test2witness.py")
        output_path = os.path.join("output", "witness.graphml")

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if property_file is None:
            property_file = resolve_path("properties", "sv-comp-reachability.prp")
        
        t2w_result = execute([
            "python3", executable, program_path, test_case,
            "--machine_model", "m64" if data_model == "LP64" else "m32",
            "--spec", property_file,
            "--producer", self.tool_name,
            "--output", output_path,
        ])

        if "Success." not in t2w_result.output: return self._abort(t2w_result)

        print(t2w_result.output)

        return output_path

    def _abort(self, result):
        print(result.err_output.decode("utf-8"))
        print(result.output)
        print("Abort.")
        return False

    def __call__(self, program_path, data_model = "LP64", cputime = None, memory = None, property_file = None, witness = False):
        ctester_executable = resolve_path("lib", "python", "ctesters", "ctesters.py")
        
        cmd = self._build_cmd(
            ctester_executable, program_path,
            data_model = data_model,
            cputime    = cputime,
            memory     = memory,
            property_file = property_file
        )

        # Execute tester
        result = execute(cmd)

        if "false(unreach-call)" not in result.output: return self._abort(result)

        test_case = find_test_case()
        if test_case is None: return self._abort(result)

        print(result.output)
        
        witness = witness or self.witness
        if not witness : return test_case

        return self._gen_witness(
            program_path,
            test_case,
            data_model = data_model,
            property_file = property_file
        )
       
    
    def __repr__(self):
        return f"{self.tool_name} [{self.version}]"


klee = Tester("klee", version = "s3")