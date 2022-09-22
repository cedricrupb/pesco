import time
import subprocess
import logging
from collections import namedtuple

ExecutionResult = namedtuple("ExecutionResult", ["returncode", "output", "err_output", "got_aborted", "wall_time"])

def execute(command, timelimit=None):
    def shut_down(process):
        process.kill()
        return process.wait()

    print(command)
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
    wall_time = time.perf_counter() - wall_time_start

    try:
        output = output.decode() if output else ""
    except UnicodeDecodeError as e:
        # fail silently, continue with encoded output
        logging.info(e, exc_info=True)

    return ExecutionResult(
        returncode, output, err_output, got_aborted, wall_time
    )

