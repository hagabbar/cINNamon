#######################################################################################################################

# -- Example Code for using the Variational Inference for Computational Imaging (VICI) Model --

#######################################################################################################################

import numpy as np
import tensorflow as tf
import scipy.io as sio
import scipy.misc as mis
import h5py
from sys import exit

from Models import VICI_forward_model
from Models import VICI_inverse_model
from Models import CVAE
from Neural_Networks import batch_manager
#from Observation_Models import simulate_observations
from data import make_samples, chris_data
import plots

run_label='gpu1',            # label for run
plot_dir="D:/LIGO/cINNamon_output/VICI/%s" % run_label,                 # plot directory

# Defining the list of parameter that need to be fed into the models
def get_params():
    params = dict(
        image_size = [1,16], # Images Size
        print_values=True, # optionally print values every report interval
        n_samples = 5000, # number of posterior samples to save per reconstruction upon inference 
        num_iterations=100001, # number of iterations inference model (inverse reconstruction)
        initial_training_rate=0.0001, # initial training rate for ADAM optimiser inference model (inverse reconstruction)
        batch_size=100, # batch size inference model (inverse reconstruction)
        report_interval=500, # interval at which to save objective function values and optionally print info during inference training
        z_dimension=800, # number of latent space dimensions inference model (inverse reconstruction)
        n_weights = 2500, # number of dimensions of the intermediate layers of encoders and decoders in the inference model (inverse reconstruction)
        save_interval=500, # interval at which to save inference model weights
        num_iterations_fw= 200001, # number of iterations of multifidelity forward model training
        initial_training_rate_fw=0.0002, # initial training rate for ADAM optimiser of multifidelity forward model training
        report_interval_fw=500, # interval at which to save objective function values and optionally print info during multifidelity forward model training
        z_dimensions_fw = 10, # latent space dimensionality of forward model
        n_weights_fw = 3000, # intermediate layers dimensionality in forward model neural networks
        batch_size_fw=100, # batch size of multifidelity forward model training
        save_interval_fw=500, # interval at which to save multi-fidelity forward model weights

        r = 4,                      # the grid dimension for the output tests
        ndim_x=3,                    # number of parameters to PE on
        sigma=0.2,                   # stadnard deviation of the noise on signal
        usepars=[0,1,2],             # which parameters you want to do PE on
        tot_dataset_size=int(2**20), # total size of training set
        ndata=16,                    # y dimension size
        seed=42,                     # random seed number
        run_label=run_label,            # label for run
        plot_dir=plot_dir,                 # plot directory
        parnames=['A','t0','tau','phi','w']    # parameter names
    )
    return params 

# You will need two types of sets to train the model: 
#
# 1) High-Fidelity Set. Small set, but with accurate measurements paired to the groundtruths. Will need the following sets:
#    - x_data_train_h = Ground-truths for which you have accurate mesurements (point estimates on pars)
#    - y_data_train_h = Accurate mesurements corresponding to x_data_train_h (noisy waveforms)
#    - y_data_train_lh = Inaccurate mesurements corresponding to x_data_train_h (noise-free waveforms)
#
# 2) Low-Fidelity Set. Large set, but without accurate measurements paired to the groundtruths. Will need the following sets:
#    - x_data_train = Ground-truths for which you only have inaccurate mesurements (point estimates on pars) 
#    - y_data_train_l = Inaccurate mesurements corresponding to x_data_train (noise-free waveforms)
#
# To run the model once it is trained you will need:
#
# y_data_test_h - new measurements you want to infer a solution posterior from    
#
# All inputs and outputs are in the form of 2D arrays, where different objects are along dimension 0 and elements of the same object are along dimension 1


# Get the training/test data and parameters of run
params=get_params()
x_data_train_h, _, y_data_train_lh, y_data_test_h,pos_test = make_samples.get_sets(params)
x_data_train, y_data_train_l, y_data_train_h = x_data_train_h, y_data_train_lh, y_data_train_lh


# Get mcmc samples
samples = chris_data.mcmc_sampler(params['r'],params['n_samples'],params['ndim_x'],y_data_test_h,params['sigma'],params['usepars'])

# First, we learn a multi-fidelity model that lerns to infer high-fidelity (accurate) observations from trget images/objects and low fidelity simulated observations. for this we use the portion of the training set for which we do have real/high fidelity observations.
#x_data_train_h = x_data_train_h.reshape(x_data_train_h.shape[0],1,x_data_train_h.shape[1])
#y_data_train_h = y_data_train_h.reshape(y_data_train_h.shape[0],1,y_data_train_h.shape[1])
#y_data_train_lh = y_data_train_lh.reshape(y_data_train_lh.shape[0],1,y_data_train_lh.shape[1])
_, _ = VICI_forward_model.train(params, x_data_train_h, y_data_train_h, y_data_train_lh, "forward_model_dir/forward_model.ckpt") # This trains the forward model and saves the weights in forward_model_dir/forward_model.ckpt

# We then train the inference model using all training images and associated low-fidelity (inaccurate) observations. Using the previously trained forward model to draw from the observation likelihood.
_, _ = VICI_inverse_model.train(params, x_data_train, y_data_train_l, np.shape(y_data_train_h)[1], "forward_model_dir/forward_model.ckpt", "inverse_model_dir/inverse_model.ckpt") # This trains the inverse model to recover posteriors using the forward model weights stored in forward_model_dir/forward_model.ckpt and saves the inverse model weights in inverse_model_dir/inverse_model.ckpt 

# The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
xm, xsx, XS, pmax = VICI_inverse_model.run(params, y_data_test_h, np.shape(x_data_train)[1], "inverse_model_dir/inverse_model.ckpt") # This runs the trained model using the weights stored in inverse_model_dir/inverse_model.ckpt
# The outputs are the following:
# - xm = marginal means
# - xsx = marginal standard deviations
# - XS = draws from the posterior (3D array with different samples for the same input along the third dimension)
# - pmax = approximate maxima (approximate 'best' reconstructions)

# Make directory for plots
plots.make_dirs(params['plot_dir'][0])

# Generate final results plots
plots.make_plots(params,samples,XS,pos_test)
