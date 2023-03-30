from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(name='taxi_env',
      version='0.0.1',
      install_requires=requirements,
      python_requires='>=3',
      packages=find_packages(),
)
