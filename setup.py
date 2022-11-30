from setuptools import setup, find_packages

setup(
  name='expplot',
  version='0.1.0',
  author='E K',
  author_email='jettabebetta@gmail.com',
  packages=['expplot'],
  # scripts=['bin/script1','bin/script2'],
  # url='http://pypi.python.org/pypi/PackageName/',
  license='LICENSE.txt',
  description='Experiment plot with confidence intervals',
  long_description=open('README.md').read(),
  install_requires=[
    'pandas==1.3.5',
    'numpy==1.21.6',
    'matplotlib==3.5.1',
    'seaborn==0.12.1',
    'scipy==1.6.0'
  ],
)