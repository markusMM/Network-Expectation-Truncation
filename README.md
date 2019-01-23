# Network-Expectation-Truncation
In contribution with the Machine Learning group at the University of Oldenburg (Olb) Germany. 
authors:
Markus Meister <meister.markus.mm@gmx.net>
Jörg Bornschein

  NET - Network Truncation Maximization
  
This is a modified and highly optimized version of the Expectation Truncation algorithm by adding a multi-label ff-Perceptron to the pobabilistic sparse coding component analysis framework.

For simplifications, here, we just use the Multi-Layer Perceptron class from SciKit-Learn for our network.

# Co-Training

In this framework, I implemented a co-training of the two models:
  - probabilistic Binary Sparse Coding
  - deterministic Multi-Layer Perceptron

While the generative model always tries to model new data points given its ifered paramters, the ff-network infers the most relevant latent variables. 
The f-network recomments those latents for the generative model for a reduced permutation of binary hidden states.

Here, the ff-network is only trained on data points generated from the generative model.
However, due to this co-training, the feed-fprward prediction seems to be as good as the expectation values of the generative model infering the hidden latent variable space.

# Example 1: Gabor-Features from 200K+ 24x24 Van'Hateren natural Image Patches

Some of the results, infering Garbor features from 200K 24x24 image patches out of the Van'Hateren nat. image database, can be seen in the figures below.

![Percepron Features](https://raw.githubusercontent.com/markusMM/Network-Expectation-Truncation/master/plots/BSC_NET_VanHateren_Gabors/W/it79.png)

*Learned Binary Sparse Coding generative fields from Co-Training Binary Sparse Coding and a Multi-Layer Perceptron. We can clearly see Gabor-like structures here.*

![Percepron Features](https://raw.githubusercontent.com/markusMM/Network-Expectation-Truncation/master/plots/BSC_NET_VanHateren_Gabors/Wperc0_1.png)

*Learned first-layer Perceptron activations from Co-Training Binary Sparse Coding and a Multi-Layer Perceptron. The probabilistic component analysis did learn Gabor-like components, similar to the receptive fiels measured in the V1-cells of the brain.*

# Example 2: Learning Edges from MNIST Data

In this example I ran the supposed graphical model on the MNIST training set. Learning a large "over-complete" feature set for many itereatings. Here, we also have used a special form of linear annealing unsing univariate uniform Gaussian data noise (with no mean) which slowily decays in slowly until 8ß% full maximum EM epochs. In the following figures we see a few of the results.

![Percepron Features](https://raw.githubusercontent.com/markusMM/Network-Expectation-Truncation/master/plots/BSC_NET_MNIST/Wperc0_1.png)

*Learned first-layer Perceptron activations from Co-Training Binary Sparse Coding and a Multi-Layer Perceptron. The probabilistic component analysis did learn Gabor-like components, similar to the receptive fiels measured in the V1-cells of the brain.*

# Usage 'API'

To use this model, you can simply run the main script with the parameter file.

(mpirun / srun [-n 4, etc]) python <main-script-name.py> <params-file-name.py>


This doc is WIP.

please be patient for further updates!
