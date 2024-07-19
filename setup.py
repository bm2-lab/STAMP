from setuptools import setup, find_packages
from os import path
from io import open

# get __version__ from _version.py
ver_file = path.join('stamp', 'version.py')
with open(ver_file) as f:
    exec(f.read())

this_directory = path.abspath(path.dirname(__file__))

# read the contents of README.md
def readme():
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        return f.read()

# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='cell-stamp',
    version=__version__,
    description='STAMP for genetic perturbation prediction',
    long_description=readme(), 
    long_description_content_type='text/markdown',
    url='https://github.com/bm2-lab/STAMP',
    author='Yicheng Gao, Zhiting Wei, Qi Liu',
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    install_requires=requirements,
    license='GPL-3.0 license'
)