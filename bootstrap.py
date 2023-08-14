# Enthought product code
#
# (C) Copyright 2010-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This file and its contents are confidential information and NOT open source.
# Distribution is prohibited.

import argparse
import subprocess
from pathlib import Path 

# Config
ROOT_DIR = Path(__file__).parent.resolve() 
ENV_NAME = "stoned-selfies"
PY_VER = 3.8
EDM_DEPS = [
    "pip",
    "setuptools"
]

PIP_DEPS = [
] 

def bootstrap(ci_mode, force=False):
    """Create and populate dev env.

    Will automatically activate the environment, unless ci_mode is True.
    Setting `force` to True will rebuild the environment from scratch. 
    """

    if ENV_NAME not in _list_edm_envs() or force is True: 
        print(f"Creating development environment {ENV_NAME}...")
        cmd = ["edm", "envs", "create", ENV_NAME, "--version", f"{PY_VER}", "--force"]
        subprocess.run(cmd, check=True)

        cmd = ["edm", "install", "-e", ENV_NAME, "-y"] + EDM_DEPS
        subprocess.run(cmd, check=True)

        # install source code pip deps
        cmd = ["edm", "run", "-e", ENV_NAME, "--", "pip", "install", "-r", "requirements.txt"]
        subprocess.run(cmd, check=True)

        # # install deploy pip deps
        # cmd = ["edm", "run", "-e", ENV_NAME, "--", "pip", "install"] + PIP_DEPS
        # subprocess.run(cmd, check=True)

        # install local project code
        cmd = ["edm", "run", "-e", ENV_NAME, "--", "pip", "install", "-e", "."]
        subprocess.run(cmd, check=True)

        print("Bootstrap complete.")

    else:
        print("Environment already exists; reusing.")

    if not ci_mode:
        print(f"Activating dev environment {ENV_NAME}")
        subprocess.run(["edm", "shell", "-e", ENV_NAME])

def _read_pip_deps(filepath=None):
    """Return the contents of the `requirments.txt` file."""
    with open(filepath, 'r') as fh: 
        contents = fh.read().splitlines()
        return contents
    
def _list_edm_envs():
    cmd = ["edm", "envs", "list"]
    proc = subprocess.run(
        cmd, check=True, capture_output=True, encoding="utf-8", errors="ignore"
    )
    envs = []
    for line in proc.stdout.split("\n"):
        parts = line.split()
        if len(parts) < 6:
            continue
        if parts[0] == "*":
            envs.append(parts[1])
        else:
            envs.append(parts[0])
    return envs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()
    bootstrap(ci_mode=args.ci, force=args.force)
