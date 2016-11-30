#!/usr/bin/env python

from setuptools import setup
from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=False)
install_reqs_parsed = [str(ir.req) for ir in install_reqs]
dep_links = [str(req_line.url) for req_line in install_reqs]

dev_reqs = parse_requirements('requirements-dev.txt', session=False)
dev_reqs_parsed = [str(ir.req) for ir in install_reqs]

setup(
    name='bb_tracking',
    version='0.11',
    description='BeesBook Tracking Utilities',
    author='Benjamin Rosemann',
    author_email='benjamin.rosemann@fu-berlin.de',
    url='https://github.com/BioroboticsLab/bb_tracking/',
    install_requires=install_reqs_parsed,
    dependency_links=dep_links,
    extras_require={'develop': dev_reqs_parsed},
    packages=['bb_tracking', 'bb_tracking.data', 'bb_tracking.tracking', 'bb_tracking.validation'],
)
