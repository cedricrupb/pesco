import os
import yaml

import logging
import zipfile

import importlib.util
import sys

import signal
import subprocess

from utils import main, download

import benchexec.result
from benchexec.tools.template import BaseTool2
from benchexec.runexecutor    import RunExecutor

logger = logging.getLogger(__name__)

LOCAL_PATH = os.path.join(os.path.dirname(__file__), "cache")

# API method ----------------------------------------------------------------

def test(tool_name : str, 
        program_path : str, 
        version : str = None,
        data_model : str = "LP64",
        cputime : int = None,
        memory  : int = None,
        container : bool = False,
        enforce_limits : bool = False,
        property_file : str = "properties/coverage-error-call.prp",
        tool_directory : str = None):
    """Help"""

    tool = load_tool(tool_name, version)
    tool.update({"container": container, "enforce_limits": enforce_limits})

    logger.info(f"Init tool {tool} ...")

    tool_directory = tool_directory or LOCAL_PATH
    tool.init(_resolve_path(tool_directory))

    logger.info(f"Run {tool} on task {program_path} ...")
    property_file = _resolve_path(property_file)

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
    tool_path = _resolve_path(tool_path)
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

        self.container = False
        self.enforce_limits = False

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
            assert len(version)  >= 0, "No version with name %s is specified" % version
            assert len(versions) == 1, "Only one version with name '%s' is allowed" % version
        
        assert len(versions) > 0, "No version is specified"
        current_version = versions[0]
        del config["archives"]
        config.update(current_version)

        config["tool_name"] = config["actor_name"]
        del config["actor_name"]

        return TestTool(**config)


    def _download_tool(self, base_dir = None):
        if base_dir is None: base_dir = LOCAL_PATH
        location_to_tool = os.path.join(base_dir, self.tool_name, self.version)
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


    def _container_exec(self, cmdline, timelimit = None, memory = None):
        executor = RunExecutor()

        def stop_run(signum, frame):
            executor.stop()
        
        signal.signal(signal.SIGINT, stop_run)

        output = "output.txt"

        result = executor.execute_run(
            args = cmdline,
            output_filename = output,
            softtimelimit = timelimit,
            memlimit = memory
        )

        if not os.path.exists(output): return Run([])

        with open(output, "r") as i:
            result = Run(i.readlines())
        
        os.remove(output)
        return result

    def _execute(self, cmdline, timelimit = None, memory = None):
        if " " in cmdline[0]: cmdline[0] = cmdline[0].replace(" ", "\ ")
        
        if self.container: return self._container_exec(cmdline, timelimit, memory)

        if self.enforce_limits:
            if timelimit is not None:
                cmdline = ["timeout", f"{timelimit}s"] + cmdline

            if memory is not None:
                cmdline = ["ulimit", "-Sv", str(int(0.9 * memory / 1024)), "&&"] + cmdline

        return execute(cmdline)


    def init(self, base_dir = None):
        if not os.path.exists(self.location):
            self._download_tool(base_dir)

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
        print("Execute the following command: %s" % " ".join(cmdline))

        run = self._execute(cmdline, timelimit = rlimits.cputime, memory = rlimits.memory)
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
    def __init__(self, output, was_timeout=False):
        self.output = output
        self.was_timeout = was_timeout

class CustomList(list):
    def any_line_contains(self, keyword):
        for line in self:
            if keyword in line:
                return True
        return False

    #def printCustom(self):
    #    for line in self:
    #        print(line)


def execute(cmdline):
    p = subprocess.Popen(" ".join(cmdline), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell = True)
    s_out, s_err = p.communicate()
    lines = CustomList([])

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

    #print("lines: ")
    #print(len(lines))
    #lines.printCustom()
    if "Timed out" in lines:
        lines.was_timeout = True

    return Run(lines)

# Path helper -------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def _resolve_path(path):
    if os.path.exists(path): return path
    return os.path.join(BASE_DIR, path)

# Runner -------------------------------------------------------------------

if __name__ == "__main__":
    main(test)