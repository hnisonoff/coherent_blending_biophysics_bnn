Code for Coherent blending of biophysics-based knowledge with Bayesian neural networks for robust protein property prediction

The scripts to reproduce the experiments in the paper are:
1) funcprior/bin/gb1.py
1) funcprior/bin/gfp.py
1) funcprior/bin/solubility.py


Dependencies:
- pytorch
- tensorflow
- deepchem
- PyTDC
- gpytorch
- seaborn
- numpy 
- pandas

Installing Dependencies:
To create a conda environment with all the required dependencies run:
    `conda env create -f environment.yml`
Note: You may need to modify the cuda versions for pytorch and tensorflow.

Once the dependencies are installed, install the provided funcprior package:
    `cd funcprior`
    `pip install .`

Datasets:
The GB1 dataset provided is from https://data.caltech.edu/records/g58c2-zzb91 under an MIT license. 
The GFP dataset provided is from https://github.com/gitter-lab/nn4dms/tree/master/data under an MIT license
The solubility dataset is provided by the Therapeutics Data Commons (https://tdcommons.ai) under an MIT license. The original data was AqSolDB and was provided under a CCO 1.0 license.
