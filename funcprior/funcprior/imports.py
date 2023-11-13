from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Normal, Bernoulli, Categorical

import gpytorch
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, IndependentMultitaskVariationalStrategy
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood, SoftmaxLikelihood

from itertools import product
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")
sns.set_context("talk")
