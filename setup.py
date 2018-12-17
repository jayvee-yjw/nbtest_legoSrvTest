#!/usr/bin/env python
from setuptools import setup, find_packages
import nbtest_legoSrvTest

with open("README.md", "r") as fh:
    long_description = fh.read()

desc = """nbtest_legoSrvTest
========

that supports Python 2.7
Usage
'''''
now just used by myself

"""

setup(
    name = 'nbtest_legoSrvTest',
    packages = ['nbtest_legoSrvTest'],
    version = nbtest_legoSrvTest.__version__,
    description = 'nbtest_legoSrvTest',
    long_description = desc,
    author = 'jayvee-yjw',
    author_email = 'gmkingyao001@gmail.com',
    url = 'https://github.com/jayvee-yjw/nbtest_legoSrvTest',
    download_url = 'https://github.com/jayvee-yjw/nbtest_legoSrvTest/archive/master.zip',
    keywords=["nbtest_legoSrvTest"],
    zip_safe=True,
    install_requires=["DictObject", "nbtest", "flask"]
)