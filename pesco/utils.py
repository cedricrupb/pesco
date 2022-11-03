import os
import time
import subprocess
import logging
from collections import namedtuple

ExecutionResult = namedtuple("ExecutionResult", ["returncode", "output", "err_output", "got_aborted", "wall_time"])

def execute(command, timelimit=None):
    def shut_down(process):
        process.kill()
        return process.wait()

    print("Execute:", command)
    logging.info(" ".join(command))

    wall_time_start = time.perf_counter()
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=False,
    ) as process:

        output = None
        err_output = None
        wall_time = None
        try:
            output, err_output = process.communicate(
                timeout=timelimit if timelimit else None
            )
            returncode = process.poll()
            got_aborted = False
        except subprocess.TimeoutExpired:
            logging.debug("Timeout of %ss expired. Killing process.", timelimit)
            returncode = shut_down(process)
            got_aborted = True
        except KeyboardInterrupt:
            logging.debug("Shutdown with CTRL-C. Killing process.")
            returncode = shut_down(process)
            got_aborted = True
            if output: print(output)
            raise

    wall_time = time.perf_counter() - wall_time_start

    try:
        output = output.decode() if output else ""
    except UnicodeDecodeError as e:
        # fail silently, continue with encoded output
        logging.info(e, exc_info=True)

    return ExecutionResult(
        returncode, output, err_output, got_aborted, wall_time
    )

# Resolve paths ---------------------------------------

from pathlib import Path

CURRENT_PATH = Path(__file__).parent.absolute()

def resolve_path(lib_path, *lib_paths):

    if len(lib_paths) > 0: lib_path = os.path.join(lib_path, *lib_paths)

    current_path = str(CURRENT_PATH)
    while "pesco" in current_path:
        target_path = os.path.join(current_path, lib_path)
        if os.path.exists(target_path): return target_path
        current_path = os.path.dirname(current_path)

    raise ValueError("Cannot find path %s in the current domain: %s" % (lib_path, CURRENT_PATH))
    
