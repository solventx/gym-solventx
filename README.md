
**Status:** Expect regular updates and bug fixes.
# OpenAI Gym environment for chemical process design and simulation

Self-assembling chemical process simulator

## Links
* Source code repository: https://github.com/solventx/gym-solventx

## Installation
You can install the module directly from github with following commands:
```
git clone https://github.com/solventx/gym-solventx.git
cd gym-solventx
pip install -e .
```

### Dependencies  
#### Graphviz
[Graphviz](https://graphviz.gitlab.io/download/). Please add Graphviz dot executable and its directory to your system's PATH variable

Example (for Windows): 
`C:\Program Files (x86)\Graphviz2.38\bin` and 
`C:\Program Files (x86)\Graphviz2.38\bin\dot.exe`

#### Cantera
Cantera is not available with pip and need to be installed within a Conda environment. You can either install in an existing environment or within a freshly created environment. Detailed instructions can be found [here.](https://cantera.org/install/conda-install.html)

Additionally, add the file **elementz.xml** located in **gym_solventx/envs/methods/dependencies/** to the Cantera sys directory. 

Example (for Windows): `C:\Users\username\AppData\Local\Continuum\anaconda3\envs\myenv\Lib\site-packages\cantera\data`

#### Other Python Modules
SciPy, Numpy, Matlplotlib, Gym, Pandas, Seaborn

## Using the module
The module can be imported as a normal python module: 

```python
import gym_solventx
```

Try out the Jupyter notebooks with a demo [here.](https://github.com/solventx/gym-solventx/blob/master/demo.ipynb)

## Unit Test
To run the provided unit test located in `gym_solventx/test/test.py` you must use an Anaconda environment since cantera is not available with pip.

The test can be run through the terminal using `pytest test.py` 

To enable Anaconda to be able to use the required dependencies from terminal you must add Anaconda to your system's PATH variable

This means adding your Anaconda Scripts directory and your anaconda python directory to Path

In windows, for example, this can be done like so:  
`setx PATH "%PATH%;C:\ProgramData\Anaconda3\Scripts;C:\ProgramData\Anaconda3"`

You will also need to initialize conda in your respective terminal:
`conda init cmd.exe` for example in windows

## Issues
Please feel free to raise an issue for bugs or feature requests.

## Who is responsible?
Core developer:  
- Blake Richey blake.e.richey@gmail.com  
- Nwike Iloeje ciloeje@anl.gov  

Contributor:  
- Siby Jose Plathottam splathottam@anl.gov 

## Acknowledgement  

## Citation
If you use this code please cite it as:

```
@misc{gym-solventx,
  title = {{gym-solventx}: Gym Environment containing a solvent extraction process simulator.},
  author = "{Blake Richey, Nwike Iloeje, Siby Jose Plathottam}",
  howpublished = {\url{https://github.com/solventx/gym-solventx}},
  url = "https://github.com/solventx/gym-solventx",
  year = 2019,
  note = "[Online; accessed 21-August-2019]"
}
```

## Copyright and License  
Copyright © 2019, UChicago Argonne, LLC

Process Design Environment Simulator for Reinforcement Learning (SolventX) is distributed under the terms of [BSD-3 OSS License.](LICENSE.md)
