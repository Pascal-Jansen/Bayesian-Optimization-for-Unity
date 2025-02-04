import numpy as np
import torch
import os
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.sampling import sample_simplex
#from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
#from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
#from botorch.fit import fit_gpytorch_model
from botorch.fit import fit_gpytorch_mll



#tkwargs = {
#    "dtype": torch.double,
#    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #"device": torch.device("cpu"),
#}

# Global Variables
#BATCH_SIZE = 1 # Number of design parameter points to query at next iteration
#NUM_RESTARTS = 10 # Used for the acquisition function number of restarts in optimization
#RAW_SAMPLES = 1024 # Durch höhere RawSamples kein OptimierungsFehler (Optimization failed within `scipy.optimize.minimize` with status 1.')
#N_ITERATIONS = 10 # Number of optimization iterations
#MC_SAMPLES = 512 # Number of samples to approximate acquisition function
#N_INITIAL = 5
#SEED = 3 # Seed to initialize the initial samples obtained


# old settings
#BATCH_SIZE = 1 # Number of design parameter points to query at next iteration
#NUM_RESTARTS = 10 # Used for the acquisition function number of restarts in optimization
#RAW_SAMPLES = 1024 # Initial restart location candidates
#N_ITERATIONS = 7 # Number of optimization iterations
#MC_SAMPLES = 512 # Number of samples to approximate acquisition function
#N_INITIAL = 3
#SEED = 2 # Seed to initialize the initial samples obtained
