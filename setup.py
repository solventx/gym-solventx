from setuptools import setup

setup(name='gym_solventx',
      version='0.0.1',
      packages=['gym_solventx',],
      description='Environment for testing Solvent Extraction configurations',
      author = 'Blake Richey',
      author_email='blake.e.richey@gmail.com',
      install_requires=['scipy>=1.2.0','numpy>=1.15.4','gym>=0.12.1', \
            'pandas>=0.24.2', 'seaborn>=0.9.0', 'graphviz>=0.11.1','pytest>=5.0.1'],	  
      )