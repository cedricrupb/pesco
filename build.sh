#!/bin/bash
#
# Build PeSCo and the environment from scratch
# for development
#
# (c) Cedric Richter, 2022
#

# Helper
function exitmsg {
    printf '%s\n' "$1" >&2 
    exit "${2-1}" 
}

# (1) Build CPAchecker

build_cpachecker() {
    CPA_PATH="lib/cpachecker/"

    patch -N "$CPA_PATH/build/build-compile.xml" < "include/cpachecker/build-compile.patch"

    pushd $CPA_PATH || exitmsg "CPAchecker does not seem to exist"

    # Run ANT
    ant || exitmsg "CPAchecker build failed"

    # Run JAR build
    ant jar || exitmsg "Something went wrong during building the CPAchecker JAR"
    
    popd

    cp -r "include/cpachecker/config" "$CPA_PATH" || exitmsg "Can copy necessary config files"

}

# (2) Download and install all Python dependencies
update_local_python_dependencies() {
    PYTHON_LIB_PATH="lib/python/"
    python3 -m pip install --upgrade -t $PYTHON_LIB_PATH -r "requirements.txt" || exitmsg "Failed to install all requirements"

    # Some libraries have to be called for intialization
    python3 init_libs.py

    rm -rf "$PYTHON_LIB_PATH/build/tree-sitter-c"

}

# (3) Download KLEE 
download_klee_artifact() {
    echo "Download KLEE"
    export PYTHONPATH="lib/python:$PYTHONPATH"
    python3 "lib/python/ctesters/ctesters.py" klee "dummy.c" --version "s3" --tool_directory "lib"
}


# Phase 1: Build CPAchecker
echo "Build CPAchecker"
build_cpachecker

# Phase 2: Python Dependencies
echo "Download Python dependencies"
update_local_python_dependencies

# Phase 3: Download KLEE
echo "Download KLEE"
download_klee_artifact
