<!--
This file is part of PeSCo,
a tool for selecting verification algorithms and executing them.

SPDX-FileCopyrightText: 2022 Cedric Richter

SPDX-License-Identifier: Apache-2.0
-->

Getting Started with PeSCo
===============================

PeSCo is a tool for selecting verification algorithms and running them. The algorithm selector decides for every given verification task, which configuration can potentially
solve the task. If a configuration is identified, PeSCo executes the configuration based on CPAchecker.

Executing PeSCo on a new task can be done as follows:
````
scripts/pesco [ --spec <SPEC_FILE> ] <SOURCE_FILE>
````
Here, a specification file needs to be given. Note that PeSCo unfolds its full 
potential only for reachability specifications. 

Building PeSCo from Sources
----------------------------
For building PeSCo from source, you must first clone the repository:
````
git clone https://github.com/cedricrupb/pesco
````
Then, run the `build.sh` script to compile PeSCo:
````
cd pesco
./build.sh
````
The script will download all necessary libraries and
build them (found under `lib`).
Note that some components are prebuild (and not compiled from scratch).
Therefore, PeSCo will mostly work on Ubuntu / Linux Systems for which PeSCo is compiled.

After the build process, the repository should be self-contained.

License and Copyright
---------------------
PeSCo is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0) and authored by [Cedric Richter](https://uol.de/en/computingscience/groups/formal-methods/team/cedric-richter). All changes
mainly address the configuration frontend. The underlying tools
belong the respective authors. Licenses can be found in the respective folder.


Prepare Programs for Verification by PeSCo
-----------------------------------------------

All programs need to pre-processed with the C pre-processor,
i.e., they may not contain #define and #include directives.

PeSCo is able to parse and analyze a large subset of (GNU)C.
If parsing fails for your program, please create an issue on
https://github.com/cedricrupb/pesco.
