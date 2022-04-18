import os
from setuptools import find_packages, setup

VERSION          = '0.0.1'
DESCRIPTION      = 'This package holds some of the communication signal processing methods I found usefull and informative'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="CommDspy",
    version=VERSION,
    author="Tomer Geva",
    author_email="<tomerg777@gmail.com>",
    description=DESCRIPTION,
    long_description=read('README.md'),
    license=read('LICENSE'),
    packages=find_packages(),
    install_requires=[],
    keywords=['python', 'signal', 'processing', 'communication', 'pam'],
    setup_requires=['pytest-runner', 'numpy', 'scipy'],
    tests_require=['pytest==6.0.1'],
    test_suite='tests',
    classifiers=[
    'Development Status :: 4 - Beta'
    ]
)