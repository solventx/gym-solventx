
**Status:** Expect regular updates and bug fixes.
# OpenAI Gym environment for chemical process design and simulation

Self-assembling chemical process simulator

## Links
* Source code repository: https://github.com/solventx/gym-solventx

## Installation
You can install the module directly from github with following commands:
```
git clone https://github.com/solventx/gym-solventx
cd gym-solventx
pip install -e .
```

## Dependencies  
You will need graphviz installed and added to your PATH variable.
[Install Graphviz](https://graphviz.gitlab.io/download/)  
Add Graphviz dot executable and its directory to your system's PATH variable

For Example: `C:\Program Files (x86)\Graphviz2.38\bin` and 
`C:\Program Files (x86)\Graphviz2.38\bin\dot.exe` for Windows.

Cantera is required to run the provided solvent extraction code.  
**Add cantera to an existing conda environment**  
`activate your_env_name`  
`conda install -c cantera cantera`  

Additionally, located in gym_solventx/envs/methods/dependencies/ is a file called 'elementz.xml'.  
if you wish to use the solvent extraction code provided, this must be added to your cantera sys directory.  
For Example: `C:\Users\username\AppData\Local\Continuum\anaconda3\envs\tf\Lib\site-packages\cantera\data`

Modules: SciPy, Numpy, Matlplotlib, Gym, Pandas, Seaborn, Graphviz

## Using the module
The module can be imported as a normal python module: import gym_solventx

Check out a demonstration with Jupyter Notebooks: [Here](https://github.com/solventx/gym-solventx/demo.ipynb)

If you need to add your Ipython Kernal from an existing conda environment:  
`conda activate my_env`  
`python -m ipykernal install --user` #Adds kernal to Jupyter Notebook/Lab in existing conda environment

## Unit Test
To run the provided unit test located in `gym_solventx/test/test.py` you must use an Anaconda environemnt since cantera is not available with pip.  

The test can be run through the terminal using `pytest test.py` 

To enable Anaconda to be able to use the required dependencies from terminal you must add Anaconda to your system's PATH variable

This means adding your Anaconda Scripts directory and your anaconda python directory to Path

In windows, for example, this can be done like so:  
`setx PATH "%PATH%;C:\ProgramData\Anaconda3\Scripts;C:\ProgramData\Anaconda3"`

You will also need to initialize conda in your respective terminal:
`conda init cmd.exe` for example in windows

## Issues
Please feel free to raise an issue when bugs are encountered or if you are need further documentation.

## Who is responsible?
Core developer:  
- Blake Richey blake.e.richey@gmail.com  
- Nwike Iloeje ciloeje@anl.gov  

Contributor:  
- Siby Jose Plathottam splathottam@anl.gov 

## Acknowledgement  

## Citation
If you use this code please cite it as:

@misc{pvder,
  title = {{gym-solventx}: Gym Environment containing a solvent extraction process simulator.},
  author = "{Blake Richey, Nwike Iloeje, Siby Jose Plathottam}",
  howpublished = {\url{https://github.com/solventx/gym-solventx}},
  url = "https://github.com/solventx/gym-solventx",
  year = 2019,
  note = "[Online; accessed 21-August-2019]"
}

## Copyright and License  
Copyright © 2019, UChicago Argonne, LLC

Process Design Environment Simulator for Reinforcement Learning (SolventX) is distributed under the terms of BSD-3 OSS License.
