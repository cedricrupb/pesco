import os
import yaml

import logging
import zipfile

import importlib.util
import sys

import subprocess

from utils import main, download

import benchexec.result
from benchexec.tools.template import BaseTool2

logger = logging.getLogger(__name__)

LOCAL_PATH = ".testers"

# API method ----------------------------------------------------------------

def test(tool_name : str, 
        program_path : str, 
        version : str = None,
        data_model : str = "LP64",
        cputime : int = None,
        memory  : int = None,
        property_file : str = "properties/coverage-error-call.prp"):
    """Help"""

    tool = load_tool(tool_name, version)
    logger.info(f"Init tool {tool} ...")
    tool.init()

    logger.info(f"Run {tool} on task {program_path} ...")
    result = tool.test(program_path, 
                        data_model = data_model,
                        cputime = cputime, 
                        memory = memory,
                        property_file = property_file)

    if result == benchexec.result.RESULT_UNKNOWN:
        print("Result: UNKNOWN, tester stopped before finishing.")
    elif result == benchexec.result.RESULT_DONE:
        print("Result: DONE, tester explored all executable paths.")
    elif result == benchexec.result.RESULT_FALSE_REACH:
        print("Result: false(unreach-call), tester found an executable path to an error location.")
        print("Results can be found in test-suite/")
    else:
        print(result)
    
    return result


# Prepare tool -------------------------------------------------------------

def load_tool(tool_name_or_path, version = None):
    if os.path.exists(tool_name_or_path):
        return TestTool.from_yaml(tool_name_or_path, version)

    tool_path = os.path.join("testers", "%s.yml" % tool_name_or_path)
    if os.path.exists(tool_path):
        return TestTool.from_yaml(tool_path, version)

    raise NotImplementedError(f"{tool_name_or_path} is not available")


class TestTool:

    def __init__(self, tool_name, **kwargs):

        self.tool_name = tool_name
        self.toolinfo_module = None

        self.version = None
        self.location = None

        self.options = []

        self.cputime = None
        self.memory  = None

        self.update(kwargs)

    def update(self, kwargs):
        for key, val in kwargs.items():
            if not hasattr(self, key): raise ValueError("Key '%s' not found" % key)
            setattr(self, key, val)

    @staticmethod
    def from_yaml(yaml_file, version = None):
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        
        versions = config["archives"]

        if version is not None:
            versions = [v for v in versions if v["version"] == version]
            assert len(versions) == 1, "Only one version with name '%s' is allowed" % version
        
        assert len(versions) > 0, "No version is specified"
        current_version = versions[0]
        del config["archives"]
        config.update(current_version)

        config["tool_name"] = config["actor_name"]
        del config["actor_name"]

        return TestTool(**config)


    def _download_tool(self):
        location_to_tool = os.path.join(LOCAL_PATH, self.tool_name, self.version)
        if os.path.exists(location_to_tool):
            self.location = location_to_tool
            return

        os.makedirs(location_to_tool)

        file_ending = self.location.split("/")[-1].split(".")[1:]
        file_ending = ".".join([""]+file_ending)
        archive_path = os.path.join(location_to_tool, "archive" + file_ending)

        if not os.path.exists(archive_path):
            download(self.location, archive_path)
        
        assert os.path.exists(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(location_to_tool)
        os.remove(archive_path)

        self.location = location_to_tool


    def _download_toolinfo(self):
        toolinfo_py = self.toolinfo_module.split("/")[-1]
        local_path  = os.path.join(self.location, toolinfo_py)
        if os.path.exists(local_path): self.toolinfo_module = local_path; return

        download(self.toolinfo_module, local_path)
        self.toolinfo_module = local_path


    def init(self):
        if not os.path.exists(self.location):
            self._download_tool()

        if not os.path.exists(self.toolinfo_module):
            self._download_toolinfo()

        assert os.path.exists(self.location) and os.path.exists(self.toolinfo_module)

        # Init tool info
        spec = importlib.util.spec_from_file_location(self.tool_name, self.toolinfo_module)
        tool_spec = importlib.util.module_from_spec(spec)
        sys.modules[self.tool_name] = tool_spec
        spec.loader.exec_module(tool_spec)

        self._tool_module_obj = tool_spec.Tool()
        self._executable = self._tool_module_obj.executable(
            BaseTool2.ToolLocator(os.path.join(self.location, self.tool_name), False, False)
        )

    def test(self, program_path, 
                data_model = "LP64", 
                property_file = "properties/coverage-error-call.prp",
                cputime = None,
                memory  = None,
                options = None):

        program_path  = os.path.abspath(program_path).replace(" ", "\ ")
        property_file = os.path.abspath(property_file).replace(" ", "\ ")

        task = BaseTool2.Task(
            [program_path], 
            None,
            property_file,
            {
                "language": "C",
                "data_model": data_model,
            }
        )

        rlimits = RLimits(**{
            "cputime": cputime or self.cputime,
            "memory" : memory or self.memory
        })

        cmdline = self._tool_module_obj.cmdline(self._executable, options or self.options, task, rlimits)

        if not rlimits.empty() and not rlimits.accessed:
            logger.warning("Tester does not support setting limits")

        logger.debug("Execute the following command: %s" % " ".join(cmdline))

        run = execute(cmdline)
        result = self._tool_module_obj.determine_result(run)
        
        return result


    def __repr__(self):
        result = "%s[%s]" % (self.tool_name, self.version)

        resources = []
        if self.cputime:
            resources.append("%s" % self.cputime)
        if self.memory:
            resources.append("%s" % self.memory)

        if len(resources) > 0:
            result += " (%s)" % ", ".join(resources)

        return result


# Rlimits -------------------------------

class RLimits:
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.accessed = False

    def empty(self):
        return len(self.config) == 0
    
    def __getattr__(self, name):
        self.accessed = True
        return self.config.get(name, None)

# Executor --------------------------------

class Run:
    def __init__(self, output):
        self.output = output

def execute(cmdline):
    p = subprocess.Popen(" ".join(cmdline), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell = True)
    s_out, s_err = p.communicate()
    lines = []

    try:
        for l in s_out.decode('utf-8').splitlines():
            print(l)
            lines.append(l)
    except:
        pass
    try:
        for l in s_err.decode('utf-8').splitlines():
            print(l)
            lines.append(l)
    except:
        pass

    return Run(lines)

# Runner -------------------------------------------------------------------

if __name__ == "__main__":
    main(test)