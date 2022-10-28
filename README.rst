=============================================
NeurIPS_2022-Generative_Hyper_Representations
=============================================


    Code to reproduce the results from NeurIPS 2022 paper "Hyper-Representations as Generative Models: Sampling Unseen Neural Network Weights".


This repository contains code to reproduce the experiments from NeurIPS 2022 paper "Hyper-Representations as Generative Models: Sampling Unseen Neural Network Weights".
The package to train and sample hyper-representations can be installed with `pip3 install -e .`.
Jupyter notebook examples are provided in `examples`.  
To download the data go to `data` and run `bash download_data.sh`.

Paper Abstract
====
Learning representations of neural network weights given a model zoo is an emerging and challenging area with many potential applications from model inspection, to neural architecture search or knowledge distillation. Recently, an autoencoder trained on a model zoo was able to learn a hyper-representation, which captures intrinsic and extrinsic properties of the models in the zoo. In this work, we extend hyper-representations for generative use to sample new model weights. We propose layer-wise loss normalization which we demonstrate is key to generate high-performing models and several sampling methods based on the topology of hyper-representations. The models generated using our methods are diverse, performant and capable to outperform strong baselines as evaluated on several downstream tasks: initialization, ensemble sampling and transfer learning. Our results indicate the potential of knowledge aggregation from model zoos to new models via hyper-representations thereby paving the avenue for novel research directions. 


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
