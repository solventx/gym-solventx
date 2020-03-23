from setuptools import setup

setup(name='gym_solventx',
      version='0.0.1',
      packages=['gym_solventx',],
      description='Environment for testing Solvent Extraction configurations',
      author = 'Siby Plathottam, Blake Richey',
      author_email='sibyjackgrove@gmail.com',
      install_requires=['gym>=0.12.1','pytest>=5.0.1'],	  
      )