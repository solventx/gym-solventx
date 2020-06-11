**Status:** Expect regular updates and bug fixes.

# OpenAI Gym environment for solvent extraction process design and simulation

Learning environment for solvent extraction process design.

## Links
* Source code repository: https://github.com/solventx/gym-solventx

## Installation
You can install the package directly from GitHub with following commands:
```
git clone https://github.com/solventx/gym-solventx.git
cd gym-solventx
pip install -e .
```

### Dependencies  
[Cantera](https://cantera.org/install/conda-install.html), Gym, SciPy, Numpy

## Using the module
The module can be imported as a normal python module: 

```python
import gym_solventx
```

Try out the demo [here.](https://github.com/solventx/gym-solventx/blob/master/demo.ipynb)

## Solvent extraction process design leaderboard
### Successful design: Recovery > 0.8, Purity > 0.985

|Name| Process configuration| Episodes before solve | Purity,Recovery (mean,std dev) | Agent details | Train script |
|--------------|-----------------------|----------|----------|-------|-------|
| [Siby Jose Plathottam](https://github.com/sibyjackgrove) | Input: Nd,Pr | 2000 | Purity: 0.85,0.04           Recovery: 0.99,0.005 |PPO with RNN||

## Issues
Please feel free to raise an issue for bugs or feature requests.

## Who is responsible?
Core developer:  
- Siby Jose Plathottam splathottam@anl.gov 
- Nwike Iloeje ciloeje@anl.gov  

Contributor:  
- Blake Richey blake.e.richey@gmail.com  

## Acknowledgement  

## Citation
If you use this code please cite it as:

```
@misc{gym-solventx,
  title = {{gym-solventx}: Gym Environment containing a solvent extraction process simulator.},
  author = "{Siby Jose Plathottam, Blake Richey, Nwike Iloeje, }",
  howpublished = {\url{https://github.com/solventx/gym-solventx}},
  url = "https://github.com/solventx/gym-solventx",
  year = 2019,
  note = "[Online; accessed 21-August-2019]"
}
```

## Copyright and License  
Copyright © 2019, UChicago Argonne, LLC

Process Design Environment Simulator for Reinforcement Learning (SolventX) is distributed under the terms of [BSD-3 OSS License.](LICENSE.md)
