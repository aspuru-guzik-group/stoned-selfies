# Enthought product code
#
# (C) Copyright 2010-2023 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This file and its contents are confidential information and NOT open source.
# Distribution is prohibited.

from setuptools import find_packages, setup

PACKAGE_NAME = "stoned_selfies"
PACKAGE_DIRS = ["stoned_selfies"]
VERSION = "0.1.0"
RUNTIME_DEPS = None


def main():
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        install_requires=RUNTIME_DEPS,
        packages=find_packages(include=PACKAGE_DIRS),
    )

if __name__ == "__main__":
    main()
