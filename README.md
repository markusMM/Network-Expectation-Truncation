# Network-Expectation-Truncation
In contribution with the Machine Learning group at the University of Oldenburg (Olb) Germany. 

Creators:

Markus Meister <meister.markus.mm@gmx.net>

Jörg Bornschein <bornschein@fias.uni-frankfurt.de>    

original version: https://github.com/jbornschein/mca-genmodel [1]

## NET - Network Truncation Maximization
  
This is a modified and highly optimized version of the Expectation Truncation algorithm [1,2] by adding a multi-label ff-Perceptron to the pobabilistic sparse coding component analysis framework.

For simplifications, here, we just use the Multi-Layer Perceptron class from SciKit-Learn for our network.

## Co-Training

In this framework, I implemented a co-training of the two models:
  - probabilistic Binary Sparse Coding
  - deterministic Multi-Layer Perceptron

While the generative model always tries to model new data points given its ifered paramters, the ff-network infers the most relevant latent variables. 
The ff-network recomments those latents for the generative model for a reduced permutation of binary hidden states.

![GrModel](https://raw.githubusercontent.com/markusMM/Network-Expectation-Truncation/master/plots/GrModelsimpleSLf.png)

*Graphical Model of Netwok Expectation Truncation. The ff-network replaces hand-tailoed model-specific selection functions for the most significant latent variable space and learns from generated data from the generative model. (c.f. [3])*

The ff-network does learn from generated data from the generative model. This model learns it's parameters directly calculating expectation values from the data given its probabilistic distributions. While at the same time, the perceptron does recommend the $H'<H$ most important latent variables due to its marginal pridiction score. This replaces hand-tailored model specific selection functions from the truncation procedure. (confere [3])

### Example 1: Gabor-Features from 200K+ 24x24 Van'Hateren natural Image Patches

Some of the results, infering Garbor features from 200K 24x24 image patches out of the Van'Hateren nat. image database, can be seen in the figures below.

![BSC Features](https://raw.githubusercontent.com/markusMM/Network-Expectation-Truncation/master/plots/BSC_NET_VanHateren_Gabors/W/it79.png)

*Learned Binary Sparse Coding generative fields from Co-Training Binary Sparse Coding and a Multi-Layer Perceptron. The probabilistic component analysis did learn Gabor-like components, similar to the receptive fiels measured in the V1-cells of the brain. (e.g. [1])*

![Percepron Features](https://raw.githubusercontent.com/markusMM/Network-Expectation-Truncation/master/plots/BSC_NET_VanHateren_Gabors/Wperc0_1.png)

*Learned first-layer Perceptron activations from Co-Training Binary Sparse Coding and a Multi-Layer Perceptron. We can clearly see Gabor-like structures here.*

### Example 2: Learning Edges from MNIST Data

In this example I ran the supposed graphical model on the MNIST training set. Learning a large "over-complete" feature set for many itereatings. Here, we also have used a special form of linear annealing unsing univariate uniform Gaussian data noise (with no mean) which slowily decays in slowly until 8ß% full maximum EM epochs. In the following figures we see a few of the results.

![Percepron Features MNIST](https://raw.githubusercontent.com/markusMM/Network-Expectation-Truncation/master/plots/BSC_NET_MNIST/Wperc0_1.png)

*Learned first-layer Perceptron activations from Co-Training Binary Sparse Coding and a Multi-Layer Perceptron. the feature map does learn features similar to edge filters in the brain. However, we do see clearly some residues from the MNIST digits remain.*



## Usage 'API'

To use this model, you can simply run the main script with the parameter file.

(mpirun / srun [-n 4, etc]) python <main-script-name.py> <params-file-name.py>

### Software Dependencies
 
 - Python (>= 2.6)
 - NumPy (reasonable recent)
 - SciPy (reasonable recent)
 - pytables (reasonable recent)
 - mpi4py (>= 1.2)
 - sklearn (reasonable recent)


## Notes

This doc is WIP.

please be patient for further updates!

## References

  [1]   -   Bornschein, Jörg & Shelton, Jacquelyn & Sheikh, Abdul-Saboor. (2012). The Maximal Causes of Binary Data. 
  
  [2]   -   Jörg Lücke and Julian Eggert. 2010. Expectation Truncation and the Benefits of Preselection In Training Generative Models. J. Mach. Learn. Res. 11 (December 2010), 2855-2900.
  
  [3]   -   Shelton, Jacquelyn & Gasthaus, Jan & Dai, Zhenwen & Lücke, Jörg & Gretton, Arthur. (2014). GP-Select: Accelerating EM Using Adaptive Subspace Preselection. Neural Computation. 29. 10.1162/NECO_a_00982. 
  

