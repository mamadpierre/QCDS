# QCDS
Quantum Circuit Design Search

This repository contains the implementations of the reinforcement learning search strategy discussed in the paper, "Quantum Circuit Design Search." 
The archive version of the paper can be found at this [link](https://arxiv.org/pdf/2012.04046.pdf). This study explores search strategies for the design of the parameterized quantum circuits.

```
requirements:

PennyLane >= 0.10
Pytorch >= 1.4

```

To run the code with defualt arguments use

```
python main.py

```

The above code executes a classical controller (neural network) to find the whole design of the PQC. If you wish the controller to find only a repetitive layer of the PQC add the argument ``` --small_design```. In addition, there can be an option of having hybrid quantum-classical controller to find a layer of PQC, to do so, use 

```
python main.py --small_design --quantumController

```

Please cite our paper, if you use the code. 

```
@article{pirhooshyaran2020quantum,
	title={Quantum circuit design search},
	author={Pirhooshyaran, Mohammad and Terlaky, Tamas},
	journal={arXiv preprint arXiv:2012.04046},
	year={2020}
}

```



