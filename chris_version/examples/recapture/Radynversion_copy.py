# coding: utf-8

# ## Radynversion Training Notebook
# 
# This notebook is used to train the Radynversion neural network. It requires the following packages:
# 
# `numpy`
# 
# `scipy`
# 
# `matplotlib`
# 
# `FrEIA: https://github.com/VLL-HD/FrEIA`
# 
# `pytorch >= 0.4.1` (only tested on `0.4.1` but will probably be updated to `1.0.x` soon -- I don't forsee any problems with this).
# 
# An NVIDIA GPU with CUDA and > 2GB VRAM is strongly recommended if you are going to attempt to train a Radynversion model. With a 1050 Ti, the full 12000 epochs are trained in under a day.
# 
# The hyperparameters listed here (learning rate, loss weights etc.) have all been empirically found to work, but changing the data may well necessitate changing these.
# 
# To (re)train Radynversion this notebook can be run pretty much from top to bottom, with only a little tweaking of the hyperparameters necessary if you change the the complexity of the input data.
# 
# A lot of the heavy lifting functions are in the files `Inn2.py` and `Loss.py`.
# 
# Please forgive the massive blobs of plotting code, the same technique is used to plot the results from the inversions and is nicely tucked away in `utils.py`, most of that code organically grew in this notebook!
# 
# To (re)train the model you will also need the training data. Either look at the ridiculously named `ExportSimpleLineBlobForTraining.py` to export the required data from your own RADYN sims/move around the atmospheric nodes etc. or use our _even_ more ridiculously named training data `DoublePicoGigaPickle50.pickle` which will be made available, along with the trained for the initial release of Radynversion on Radynversion's Github releases page. The training pickle contains all of the snapshots from the Fokker-Planck RADYN simulations in the F-CHROMA grid, sampled at the 50 atmospheric points detailed in Osborne, Armstrong, and Fletcher (2019).

from Inn2 import RadynversionNet, AtmosData, RadynversionTrainer
import loss as Loss
import pickle
import numpy as np
import scipy
from scipy.stats import multivariate_normal as mvn
from scipy.special import logit, expit
from scipy.stats import uniform, norm, gaussian_kde, ks_2samp, anderson_ksamp
from scipy import stats
from scipy.signal import butter, lfilter, freqs, resample
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.utils.data
import h5py
import os, shutil
from sys import exit
import corner
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from time import time

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import rev_multiplicative_layer, F_fully_connected, F_conv
from double_network import nn_double_f, nn_double_r, DoubleNetTrainer

import chris_data as data_maker

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# global parameters
sig_model = 'sg'   # the signal model to use
usepars = [0,1,2]    # parameter indices to use
run_label='gpu1'
out_dir = "D:/LIGO/cINNamon_output/%s" % run_label
do_posterior_plots=True
ndata=16           # length of 1 data sample
ndim_x=3           # number of parameters to PE on
ndim_y = ndata     # length of 1 data sample

ndim_z = 16      # size of latent space

Ngrid=20
n_neurons = 64
ndim_tot = max(ndim_x+ndim_y,ndim_y+ndim_z) + n_neurons # 384     
r = 2             # the grid dimension for the output tests
sigma = 0.2        # the noise std
seed = 1           # seed for generating data
test_split = r*r   # number of testing samples to use
test_sample_idx=4

N_samp = 5000 # number of test samples to use after training
plot_cadence = 500  # make plots every N iterations
numInvLayers=6
dropout=0.0
batchsize=2048
filtsize = 3       # TODO
clamp=2.0          # TODO
tot_dataset_size=int(1e4) # 2**20 TODO really should use 1e8 once cpu is fixed

tot_epoch=1000000
lr=2.5e-4
zerosNoiseScale=5e-2
y_noise_scale = None

wPred=500.0        #4000.0
wLatent= 2000.0     #900.0
wRev= 2000.0        #1000.0

do_cnn = True     # if True, use cnn in double neural network
do_double_nn=True    # if True, use the double neural network model. If False, use Radyn model
do_covar=False      # if True, use a covariance loss on the forward mean squared error
review_mmd=False   # if True, turn on ideal alpha finder
extra_z = True    # if True, append y noise realization to x vector
fadeIn=True       # If True, fade in the backward MMD loss
latentAlphas=[1.0]#[0.2, 0.5, 0.9, 1.3, 2.4, 5.0, 10.0, 20.0, 40.0, 500.0, 700.0, 900.0]
backwardAlphas=None # [1.4, 2, 5.5, 7]
conv_nn = False    # Choose to use convolutional layers. TODO
multi_par=True
load_dataset=False  # use prevoiusly made training set
load_model=False  # use previously trained model
do_contours=True
gen_inf_temp=True  # generate templates on the fly
do_mcmc=True
latent=False      # if True, add color distance scale to posterior scatter plots
dataLocation1 = 'benchmark_data_%s.h5py' % run_label
T = 1.0           # length of time series (s)
dt = T/ndata        # sampling time (Sec)
fnyq = 0.5/dt   # Nyquist frequency (Hz)
if multi_par==True: bound = [0.0,1.0,0.0,1.0,0.0,1.0*fnyq,0.0,3.0,0.0,1.0]
else: bound = [0.0,1.0,0.0,1.0] # effective bound for the liklihood

def pp_plot(truth,samples):
    """
    generates the pp plot data given samples and truth values
    """
    Nsamp = samples.shape[0]
    kernel = gaussian_kde(samples.transpose())
    v = kernel.pdf(truth)
    x = kernel.pdf(samples.transpose())
    r = np.sum(x>v)/float(Nsamp)

    return r

def plot_pp(model,labels_test,pos_test,zeros_noise_scale,y_noise_scale,Nsamp,Npp,ndim_x,ndim_y,ndim_z,ndim_tot,outdir,i_epoch,conv=False):
    """
    make p-p plots
    """
    out_shape = [-1,ndim_tot]
    if conv==True:
        in_shape = [-1,1,ndim_tot]
    else:
        in_shape = [-1,ndim_tot]
    plt.figure()
    pp = np.zeros(Npp+2)
    pp[0] = 0.0
    pp[1] = 1.0
    for cnt in range(Npp):

        # convert data into correct format
        y_samps = np.tile(np.array(labels_test[cnt,:]),Nsamp).reshape(Nsamp,ndim_y)
        y_samps = torch.tensor(y_samps,dtype=torch.float,device=device)

        # make the new padding for the noisy data and latent vector data
        pad_zy = zeros_noise_scale * torch.randn(Nsamp,ndim_tot-ndim_y-ndim_z,device=device)

        # add noise to y data (why?)
        y = y_samps # + y_noise_scale * torch.randn(Nsamp,ndim_y,device=device)

        # make some z data
        z = torch.randn(Nsamp,ndim_z,device=device)

        # make a padded zy vector (with all new noise)
        zy_rev_padded = torch.cat((y,z,pad_zy),dim=1)

        # apply reverse model to the y data and original z data
        output_rev = model(zy_rev_padded.reshape(in_shape),rev=True).reshape(out_shape)
        output_rev_x = output_rev[:,:ndim_x]  # extract the model output x
        rev_x = output_rev_x.cpu().data.numpy()

        pp[cnt+2] = pp_plot(pos_test.numpy()[cnt,:],rev_x[:,:])

        plt.plot(np.arange(Npp+2)/(Npp+1.0),np.sort(pp),'-')
        plt.plot([0,1],[0,1],'--k')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.savefig('%s/pp_plot_%04d.png' % (outdir,i_epoch),dpi=360)
        plt.savefig('%s/latest/latest_pp_plot.png' % outdir,dpi=360)
        plt.close()
    return

def plot_y_evolution(model,dim,parnames,ndim_x,ndim_y,ndim_z,ndim_tot,outdir,i_epoch,conv=False,model_f=None,model_r=None,do_double_nn=False,do_cnn=False):
    """
    Plot examples of test y-data generation
    """
    n = 10
    out_shape = [-1,ndim_tot]
    if conv==True:
        in_shape = [-1,1,ndim_tot]
    else:
        in_shape = [-1,ndim_tot]
    fig, axes = plt.subplots(n,2,figsize=(6,6))

    # make a set of x parameter lines through the space
    x_orig = np.array(n*[0.5,0.5,0.5,0.2,0.2]).reshape(n,5)
    x_orig[:,dim] = (1.0/n)*np.arange(n)
    x_evo = x_orig[:,:ndim_x]

    # run the x test data through the model
    x = torch.tensor(x_evo,dtype=torch.float,device=dev)

    # make the new padding for the noisy data and latent vector data
    pad_x = torch.zeros(n,ndim_tot-ndim_x,device=dev)

    # make a padded zy vector (with all new noise)
    x_padded = torch.cat((x,pad_x),dim=1)

    # apply forward model to the x data
    if do_double_nn:
        output = model_f(torch.cat((x),dim=1))#.reshape(out_shape)
        output_y = output[:,:ndim_y]  # extract the model output y
        output_z = output[:,ndim_y:]  # extract the model output z
    else:
        output = model(x_padded.reshape(in_shape))#.reshape(out_shape)
        output_y = output[:,model.outSchema.timeseries]  # extract the model output y
        output_z = output[:,model.outSchema.LatentSpace]  # extract the model output z
    y = output_y.cpu().data.numpy()
    z = output_z.cpu().data.numpy()

    # loop over input parameters
    for i in range(10):

        # make noise free waveform for plotting
        t = np.arange(ndim_y)/float(ndim_y)
        A,t0,tau,p,w = x_orig[i,:]
        fnyq = 0.5*len(t)
        s = A*np.sin(2.0*np.pi*(w*fnyq*(t-t0) + p))*np.exp(-((t-t0)/tau)**2)

        axes[i,0].clear()
        axes[i,0].plot(np.arange(ndim_y)/float(ndim_y),y[i,:],'b-')
        axes[i,0].plot(np.arange(ndim_y)/float(ndim_y),s,'r-',alpha=0.5)
        axes[i,0].set_xlim([0,1])
        axes[i,0].set_ylim([-1.0,1.0])
        matplotlib.rc('xtick', labelsize=8)
        matplotlib.rc('ytick', labelsize=8)
        axes[i,0].set_xlabel('t') if i==n-1 else axes[i,0].set_xlabel('')
        axes[i,0].set_ylabel('y(t)')

        axes[i,1].clear()
        axes[i,1].plot(np.arange(ndim_z),z[i,:],'b-',alpha=0.5)
        axes[i,1].plot(np.arange(ndim_z),z[i,:],'b.',markersize=5)
        axes[i,1].set_xlim([0,ndim_z-1])
        axes[i,1].set_ylim([-3,3])
        matplotlib.rc('xtick', labelsize=8)
        matplotlib.rc('ytick', labelsize=8)
        axes[i,1].set_xlabel('n') if i==n-1 else axes[i,1].set_xlabel('')
        axes[i,1].set_ylabel('z')

        txt = '%s: %.2f' % (parnames[dim],x_evo[i,dim])
        axes[i,1].text(0.95, 0.95, txt,
                       fontsize=6,
                       horizontalalignment='right',
                       verticalalignment='top',
                       transform=axes[i,1].transAxes)

    fig.canvas.draw()
    fig.savefig('%s/yevo_%d_%04d.png' % (outdir,dim,i_epoch),dpi=360)
    fig.savefig('%s/latest/latest_yevo_%d.png' % (outdir,dim),dpi=360)
    plt.close(fig)
    return

def plot_z_dist(model,ndim_x,ndim_y,ndim_z,ndim_tot,usepars,sigma,outdir,i_epoch,conv=False,model_f=None,model_r=None,do_double_nn=False,do_cnn=False):
    """
    Plots the distribution of latent z variables
    """
    Nsamp = 250
    out_shape = [-1,ndim_tot]
    if conv==True:
        in_shape = [-1,1,ndim_tot]
    else:
        in_shape = [-1,ndim_tot]

    # generate test data
    x_test, y_test, x, sig_test, parnames = data_maker.generate(
        tot_dataset_size=Nsamp,
        ndata=ndim_y,
        usepars=usepars,
        sigma=sigma,
        seed=1
    )
   
    # run the x test data through the model
    x = torch.tensor(x_test,dtype=torch.float,device=dev).clone().detach()
    y_test = torch.tensor(y_test,dtype=torch.float,device=dev).clone().detach()
    sig_test = torch.tensor(sig_test,dtype=torch.float,device=dev).clone().detach()

    # make the new padding for the noisy data and latent vector data
    pad_x = torch.zeros(Nsamp,ndim_tot-ndim_x-ndim_y,device=dev)

    # make a padded zy vector (with all new noise)
    x_padded = torch.cat((x,pad_x,y_test-sig_test),dim=1)

    # apply forward model to the x data
    if do_double_nn:
        if do_cnn:
            data = torch.cat((x,y_test-sig_test), dim=1)
            output = model_f(data.reshape(data.shape[0],1,data.shape[1]))#.reshape(out_shape)
            output_z = output[:,ndim_y:]  # extract the model output y
        else:
            output = model_f(torch.cat((x,y_test-sig_test), dim=1))#.reshape(out_shape)
            output_z = output[:,ndim_y:]  # extract the model output y
    else:
        output = model(x_padded.reshape(in_shape))#.reshape(out_shape)
        output_z = output[:,model.outSchema.LatentSpace]  # extract the model output y
    z = output_z.cpu().data.numpy()
    C = np.cov(z.transpose())

    fig, axes = plt.subplots(1,figsize=(5,5))
    im = axes.imshow(np.abs(C))

    # We want to show all ticks...
    axes.set_xticks(np.arange(ndim_z))
    axes.set_yticks(np.arange(ndim_z))

    # Rotate the tick labels and set their alignment.
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(ndim_z):
        for j in range(ndim_z):
            text = axes.text(j,i,'%.2f' % C[i,j], fontsize=3,
                       ha="center",va="center",color="w")

    fig.tight_layout()
    fig.savefig('%s/cov_z_%04d.png' % (outdir,i_epoch),dpi=360)
    fig.savefig('%s/latest/latest_cov_z.png' % outdir,dpi=360)
    plt.close(fig)

    fig, axes = plt.subplots(ndim_z,ndim_z,figsize=(5,5))
    for c in range(ndim_z):
        for d in range(ndim_z):
            if d<c:
                patches = []
                axes[c,d].clear()
                matplotlib.rc('xtick', labelsize=8)
                matplotlib.rc('ytick', labelsize=8)
                axes[c,d].plot(z[:,c],z[:,d],'.r',markersize=0.5)
                circle1 = Circle((0.0, 0.0), 1.0,fill=False,linestyle='--')
                patches.append(circle1)
                circle2 = Circle((0.0, 0.0), 2.0,fill=False,linestyle='--')
                patches.append(circle2)
                circle3 = Circle((0.0, 0.0), 3.0,fill=False,linestyle='--')
                patches.append(circle3)
                p = PatchCollection(patches, alpha=0.2)
                axes[c,d].add_collection(p)
                axes[c,d].set_yticklabels([])
                axes[c,d].set_xticklabels([])
                axes[c,d].set_xlim([-3,3])
                axes[c,d].set_ylim([-3,3])
            else:
                axes[c,d].axis('off')
            axes[c,d].set_xlabel('')
            axes[c,d].set_ylabel('')

    fig.savefig('%s/scatter_z_%04d.png' % (outdir,i_epoch),dpi=360)
    fig.savefig('%s/latest/latest_scatter_z.png' % outdir,dpi=360)
    plt.close(fig)
    
    fig, axes = plt.subplots(1,figsize=(5,5))
    delta = np.transpose(z[:,:])
    dyvec = np.linspace(-10*1.0,10*1.0,250)
    for d in delta:
        plt.hist(np.array(d).flatten(),25,density=True,histtype='stepfilled',alpha=0.5)
    plt.hist(np.array(delta).flatten(),25,density=True,histtype='step',linestyle='dashed')
    plt.plot(dyvec,norm.pdf(dyvec,loc=0,scale=1.0),'k-')
    plt.xlabel('predicted z')
    plt.ylabel('p(z)') 

    fig.savefig('%s/dist_z_%04d.png' % (outdir,i_epoch),dpi=360)
    fig.savefig('%s/latest/latest_dist_z.png' % outdir,dpi=360)
    plt.close(fig)

    return

def plot_x_evolution(model,ndim_x,ndim_y,ndim_z,ndim_tot,sigma,parnames,outdir,i_epoch,conv=False,model_f=None,model_r=None,do_double_nn=False,do_cnn=False):
    """
    Plot examples of test y-data generation
    """
    Nsamp = 100
    out_shape = [-1,ndim_tot]
    if conv==True:
        in_shape = [-1,1,ndim_tot]
    else:
        in_shape = [-1,ndim_tot]
    fig, axes = plt.subplots(ndim_x,ndim_x,figsize=(6,6))

    # make a noisy signal in the middle of the space
    t = np.arange(ndim_y)/float(ndim_y)
    A,t0,tau,p,w = np.array([0.5,0.5,0.5,0.2,0.2])
    fnyq = 0.5*len(t)
    s = A*np.sin(2.0*np.pi*(w*fnyq*(t-t0) + p))*np.exp(-((t-t0)/tau)**2)

    y_orig = s + np.random.normal(loc=0.0,scale=sigma,size=ndim_y)
    y = torch.tensor(np.tile(np.array(y_orig),Nsamp+1).reshape(Nsamp+1,ndim_y),dtype=torch.float,device=dev)

    # make random colors
    cols = ['r','b','g']

    # loop over different shells of z
    for j in range(3):

        # make specific z values
        temp = np.random.normal(loc=0.0,scale=1.0,size=(Nsamp+1,ndim_z))
        z = (j+1)*np.array([t/np.linalg.norm(t) for t in temp])
        z = torch.tensor(z,dtype=torch.float,device=dev)
        pad_yz = torch.zeros(Nsamp+1,ndim_tot-ndim_y-ndim_z,device=dev)
        yz_padded = torch.cat((y,z,pad_yz),dim=1)

        # apply backward model to the padded yz data
        if do_double_nn:
            if do_cnn:
                data = torch.cat((y,z), dim=1)
                output = model_r(data.reshape(data.shape[0],1,data.shape[1]))#.reshape(out_shape)
                output_x = output[:,:ndim_x]  # extract the model output y
            else:
                output = model_r(torch.cat((y,z), dim=1))#.reshape(out_shape)
                output_x = output[:,:ndim_x]  # extract the model output y
        else:
            output = model(yz_padded.reshape(in_shape),rev=True)#.reshape(out_shape)
            output_x = output[:,model.inSchema.amp[0]:model.inSchema.tau[-1]+1]  # extract the model output y
        x = output_x.cpu().data.numpy()

        # loop over input parameters
        for i in range(ndim_x):
            for k in range(ndim_x):
                if k<i:
                    axes[i,k].plot(x[:,i],x[:,k],'.',markersize=0.5,color=cols[j])
                    axes[i,k].set_xlim([0,1])
                    axes[i,k].set_ylim([0,1])
                    matplotlib.rc('xtick', labelsize=8)
                    matplotlib.rc('ytick', labelsize=8)
                    axes[i,k].set_xlabel(parnames[i])
                    axes[i,k].set_ylabel(parnames[k])
                elif k==ndim_x-2 and i==ndim_x-2:
                    axes[i,k].plot(np.arange(ndim_y)/float(ndim_y),y_orig,'b-')
                    axes[i,k].plot(np.arange(ndim_y)/float(ndim_y),s,'r-')
                    axes[i,k].set_xlim([0,1])
                    axes[i,k].set_ylim([-1,1])
                    matplotlib.rc('xtick', labelsize=8)
                    matplotlib.rc('ytick', labelsize=8)
                    axes[i,k].set_xlabel('t')
                    axes[i,k].set_ylabel('y')
                else:
                    axes[i,k].axis('off')

    fig.canvas.draw()
    plt.savefig('%s/xevo_%04d.png' % (outdir,i_epoch),dpi=360)
    plt.savefig('%s/latest/latest_xevo.png' % (outdir),dpi=360)
    plt.close()
    return

def plot_y_dist(model,ndim_x,ndim_y,ndim_z,ndim_tot,usepars,sigma,outdir,i_epoch,conv=False,model_f=None,model_r=None,do_double_nn=False,do_cnn=False):
    """
    Plots the joint distributions of y variables
    """
    Nsamp = 1000
    out_shape = [-1,ndim_tot]
    if conv==True:
        in_shape = [-1,1,ndim_tot]
    else:
        in_shape = [-1,ndim_tot]

    # generate test data
    x_test, y_test, x, sig_test, parnames = data_maker.generate(
        tot_dataset_size=Nsamp,
        ndata=ndim_y,
        usepars=usepars,
        sigma=sigma,
        seed=1
    )

    # run the x test data through the model
    x = torch.tensor(x_test,dtype=torch.float,device=dev).clone().detach()
    y_test = torch.tensor(y_test,dtype=torch.float,device=dev).clone().detach()
    sig_test = torch.tensor(sig_test,dtype=torch.float,device=dev).clone().detach()

    # make the new padding for the noisy data and latent vector data
    pad_x = torch.zeros(Nsamp,ndim_tot-ndim_x-ndim_y,device=dev)

    # make a padded zy vector (with all new noise)
    x_padded = torch.cat((x,pad_x,y_test-sig_test),dim=1)

    # apply forward model to the x data
    if do_double_nn:
        if do_cnn:
            data = torch.cat((x,y_test-sig_test), dim=1)
            output = model_f(data.reshape(data.shape[0],1,data.shape[1]))#.reshape(out_shape)
            output_y = output[:,:ndim_y]  # extract the model output y
        else:
            output = model_f(torch.cat((x,y_test-sig_test), dim=1))#.reshape(out_shape)
            output_y = output[:,:ndim_y]  # extract the model output y    
    else:
        output = model(x_padded.reshape(in_shape))
        output_y = output[:, model.outSchema.timeseries]
    y = output_y.cpu().data.numpy()
    sig_test = sig_test.cpu().data.numpy()
    dy = y - sig_test
    C = np.cov(dy.transpose())

    fig, axes = plt.subplots(1,figsize=(5,5))

    im = axes.imshow(C)

    # We want to show all ticks...
    axes.set_xticks(np.arange(ndim_y))
    axes.set_yticks(np.arange(ndim_y))

    # Rotate the tick labels and set their alignment.
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(ndim_y):
        for j in range(ndim_y):
            text = axes.text(j,i,'%.2f' % C[i,j], fontsize=3,
                       ha="center",va="center",color="w")

    fig.tight_layout()
    plt.savefig('%s/cov_y_%04d.png' % (outdir,i_epoch),dpi=360)
    plt.savefig('%s/latest/latest_cov_y.png' % outdir,dpi=360)
    plt.close(fig)

    fig, axes = plt.subplots(1,figsize=(5,5))
    delta = np.transpose(y[:,:]-sig_test[:,:])
    dyvec = np.linspace(-10*sigma,10*sigma,250)
    for d in delta:
        plt.hist(np.array(d).flatten(),25,density=True,histtype='stepfilled',alpha=0.5)
    plt.hist(np.array(delta).flatten(),25,density=True,histtype='step',linestyle='dashed')
    plt.plot(dyvec,norm.pdf(dyvec,loc=0,scale=np.sqrt(2.0)*sigma),'k-')
    plt.xlabel('y-y_pred')
    plt.ylabel('p(y-y_pred)')
    plt.savefig('%s/y_dist_%04d.png' % (outdir,i_epoch),dpi=360)
    plt.savefig('%s/latest/y_dist.png' % outdir,dpi=360)
    plt.close(fig)
    return

    return

def result_stat_tests(inn_samps, mcmc_samps, cnt, parnames):
    """
    Record and print ks and AD test statistics
    """
    
    ks_mcmc_arr = []
    ks_inn_arr = []
    ad_mcmc_arr = []
    ad_inn_arr = []

    # iterate through each parameter
    for i in range(inn_samps.shape[1]):
        # get ideal bayesian number. We want the 2 tailed p value from the KS test FYI
        ks_mcmc_result = ks_2samp(mcmc_samps[:int(mcmc_samps.shape[0]/2.0), i], mcmc_samps[int(mcmc_samps.shape[0]/2.0):, i])
        ad_mcmc_result = anderson_ksamp([mcmc_samps[:int(mcmc_samps.shape[0]/2.0), i], mcmc_samps[int(mcmc_samps.shape[0]/2.0):, i]])

        # get predicted vs. true number
        ks_inn_result = ks_2samp(inn_samps[:,i],mcmc_samps[:,i])
        ad_inn_result = anderson_ksamp([inn_samps[:,i],mcmc_samps[:,i]])
        #print('Test Case %d, Parameter(%s) k-s result: [Ideal(%.6f), Predicted(%.6f)]' % (int(cnt),parnames[i],np.array(ks_mcmc_result[1]),np.array(ks_inn_result[1])))
        #print('Test Case %d, Parameter(%s) A-D result: [Ideal(%.6f), Predicted(%.6f)]' % (int(cnt),parnames[i],np.array(ad_mcmc_result[0]),np.array(ad_inn_result[0])))

        # store result stats
        ks_mcmc_arr.append(ks_mcmc_result[1])
        ks_inn_arr.append(ks_inn_result[1])
        ad_mcmc_arr.append(ad_mcmc_result[0])
        ad_inn_arr.append(ad_inn_result[0])

    return ks_mcmc_arr, ks_inn_arr, ad_mcmc_arr, ad_inn_arr

def plot_y_test(model,Nsamp,usepars,sigma,ndim_x,ndim_y,ndim_z,ndim_tot,outdir,r,i_epoch,conv=False,model_f=None,model_r=None,do_double_nn=False,do_cnn=False):
    """
    Plot examples of test y-data generation
    """

    # generate test data
    x_test, y_test, x, sig_test, parnames = data_maker.generate(
        tot_dataset_size=Nsamp,
        ndata=ndim_y,
        usepars=usepars,
        sigma=sigma,
        seed=1
    )

    out_shape = [-1,ndim_tot]
    if conv==True:
        in_shape = [-1,1,ndim_tot]
    else:
        in_shape = [-1,ndim_tot]
    fig, axes = plt.subplots(r,r,figsize=(6,6),sharex='col',sharey='row')

    # run the x test data through the model
    x = torch.tensor(x_test[:r*r,:],dtype=torch.float,device=dev).clone().detach()
    y_test = torch.tensor(y_test[:r*r,:],dtype=torch.float,device=dev).clone().detach()
    sig_test = torch.tensor(sig_test[:r*r,:],dtype=torch.float,device=dev).clone().detach()

    # make the new padding for the noisy data and latent vector data
    pad_x = torch.zeros(r*r,ndim_tot-ndim_x-ndim_y,device=dev)
 
    # make a padded zy vector (with all new noise)
    x_padded = torch.cat((x,pad_x,y_test-sig_test),dim=1)

    # apply forward model to the x data
    if do_double_nn:
        if do_cnn:
            data = torch.cat((x,y_test-sig_test), dim=1)
            output = model_f(data.reshape(data.shape[0],1,data.shape[1]))#.reshape(out_shape)
            output_y = output[:,:ndim_y]  # extract the model output y
        else:
            output = model_f(torch.cat((x,y_test-sig_test), dim=1))#.reshape(out_shape)
            output_y = output[:,:ndim_y]  # extract the model output y
    else:
        output = model(x_padded.reshape(in_shape))#.reshape(out_shape)
        output_y = output[:,model.outSchema.timeseries]  # extract the model output y
    y = output_y.cpu().data.numpy()

    cnt = 0
    for i in range(r):
        for j in range(r):

            axes[i,j].clear()
            axes[i,j].plot(np.arange(ndim_y)/float(ndim_y),y[cnt,:],'b-')
            axes[i,j].plot(np.arange(ndim_y)/float(ndim_y),y_test[cnt,:].cpu().data.numpy(),'k',alpha=0.5)
            axes[i,j].set_xlim([0,1])
            #matplotlib.rc('xtick', labelsize=5)
            #matplotlib.rc('ytick', labelsize=5)
            axes[i,j].set_xlabel('t') if i==r-1 else axes[i,j].set_xlabel('')
            axes[i,j].set_ylabel('y') if j==0 else axes[i,j].set_ylabel('')
            if i==0 and j==0:
                axes[i,j].legend(('pred y','y'))
            cnt += 1

    fig.canvas.draw()
    fig.savefig('%s/ytest_%04d.png' % (outdir,i_epoch),dpi=360)
    fig.savefig('%s/latest/latest_ytest.png' % outdir,dpi=360)
    plt.close(fig)
    return

def main():
    ## If file exists, delete it ##
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    else:    ## Show a message ##
        print("Attention: %s file not found" % out_dir)

    # setup output directory - if it does not exist
    os.makedirs('%s' % out_dir)
    os.makedirs('%s/latest' % out_dir)
    os.makedirs('%s/animations' % out_dir)


    # generate data
    if not load_dataset:
        pos, labels, x, sig, parnames = data_maker.generate(
            tot_dataset_size=tot_dataset_size,
            ndata=ndata,
            usepars=usepars,
            sigma=sigma,
            seed=seed
        )
        print('generated data')

        hf = h5py.File('benchmark_data_%s.h5py' % run_label, 'w')
        hf.create_dataset('pos', data=pos)
        hf.create_dataset('labels', data=labels)
        hf.create_dataset('x', data=x)
        hf.create_dataset('sig', data=sig)
        hf.create_dataset('parnames', data=np.string_(parnames))

    data = AtmosData([dataLocation1], test_split, resampleWl=None)
    data.split_data_and_init_loaders(batchsize)

    # seperate the test data for plotting
    pos_test = data.pos_test
    labels_test = data.labels_test
    sig_test = data.sig_test

    ndim_x = len(usepars)
    print('Computing MCMC posterior samples')
    if do_mcmc or not load_dataset:
        # precompute true posterior samples on the test data
        cnt = 0
        samples = np.zeros((r*r,N_samp,ndim_x))
        for i in range(r):
            for j in range(r):
                samples[cnt,:,:] = data_maker.get_lik(np.array(labels_test[cnt,:]).flatten(),sigma=sigma,usepars=usepars,Nsamp=N_samp)
                print(samples[cnt,:10,:])
                cnt += 1

        # save computationaly expensive mcmc/waveform runs
        if load_dataset==True:
            # define names of parameters to have PE done on them
           
            parnames=['A','t0','tau','phi','w']
            names = [parnames[int(i)] for i in usepars]
            hf = h5py.File('benchmark_data_%s.h5py' % run_label, 'w')
            hf.create_dataset('pos', data=data.pos)
            hf.create_dataset('labels', data=data.labels)
            hf.create_dataset('x', data=data.x)
            hf.create_dataset('sig', data=data.sig)
            hf.create_dataset('parnames', data=np.string_(names))
        hf.create_dataset('samples', data=np.string_(samples))
        hf.close()

    else:
        samples=h5py.File(dataLocation1, 'r')['samples'][:]
        parnames=h5py.File(dataLocation1, 'r')['parnames'][:]

    # plot the test data examples
    plt.figure(figsize=(6,6))
    fig, axes = plt.subplots(r,r,figsize=(6,6),sharex='col',sharey='row')
    cnt = 0
    for i in range(r):
        for j in range(r):
            axes[i,j].plot(data.x,np.array(labels_test[cnt,:]),'-', label='noisy')
            axes[i,j].plot(data.x,np.array(sig_test[cnt,:]),'-', label='noise-free')
            axes[i,j].legend(loc='upper left')
            cnt += 1
            axes[i,j].axis([0,1,-1.5,1.5])
            axes[i,j].set_xlabel('time') if i==r-1 else axes[i,j].set_xlabel('')
            axes[i,j].set_ylabel('h(t)') if j==0 else axes[i,j].set_ylabel('')
    plt.savefig('%s/test_distribution.png' % out_dir,dpi=360)
    plt.close()

    # initialize plot for showing testing results
    fig, axes = plt.subplots(r,r,figsize=(6,6),sharex='col',sharey='row')

    for k in range(ndim_x):
        parname1 = parnames[k]
        for nextk in range(ndim_x):
            parname2 = parnames[nextk]
            if nextk>k:
                cnt = 0
                for i in range(r):
                     for j in range(r):

                         # plot the samples and the true contours
                         axes[i,j].clear()
                         axes[i,j].scatter(samples[cnt,:,k], samples[cnt,:,nextk],c='b',s=0.5,alpha=0.5, label='MCMC')
                         axes[i,j].plot(pos_test[cnt,k],pos_test[cnt,nextk],'+c',markersize=8, label='MCMC Truth')
                         axes[i,j].set_xlim([0,1])
                         axes[i,j].set_ylim([0,1])
                         axes[i,j].set_xlabel(parname1) if i==r-1 else axes[i,j].set_xlabel('')
                         axes[i,j].set_ylabel(parname2) if j==0 else axes[i,j].set_ylabel('')
                         axes[i,j].legend(loc='upper left')
                         
                         cnt += 1

                # save the results to file
                fig.canvas.draw()
                plt.savefig('%s/true_samples_%d%d.png' % (out_dir,k,nextk),dpi=360)


    def store_pars(f,pars):
        for i in pars.keys():
            f.write("%s: %s\n" % (i,str(pars[i])))
        f.close()

    # store hyperparameters for posterity
    f=open("%s_run-pars.txt" % run_label,"w+")
    pars_to_store={"sigma":sigma,"ndata":ndata,"T":T,"seed":seed,"n_neurons":n_neurons,"bound":bound,"conv_nn":conv_nn,"filtsize":filtsize,"dropout":dropout,
                   "clamp":clamp,"ndim_z":ndim_z,"tot_epoch":tot_epoch,"lr":lr, "latentAlphas":latentAlphas, "backwardAlphas":backwardAlphas,
                   "zerosNoiseScale":zerosNoiseScale,"wPred":wPred,"wLatent":wLatent,"wRev":wRev,"tot_dataset_size":tot_dataset_size,
                   "numInvLayers":numInvLayers,"batchsize":batchsize}
    store_pars(f,pars_to_store)

    if extra_z: inRepr = [('amp', 1), ('t0', 1), ('tau', 1), ('!!PAD',), ('yNoise', data.atmosOut.shape[1])]
    else: inRepr = [('amp', 1), ('t0', 1), ('tau', 1), ('!!PAD',)]
    outRepr = [('LatentSpace', ndim_z), ('!!PAD',), ('timeseries', data.atmosOut.shape[1])]
    if do_double_nn:
        model_f = nn_double_f((ndim_x+ndim_y),(ndim_y+ndim_z))
        model_r = nn_double_r((ndim_y+ndim_z),(ndim_x+ndim_y))
    else:
        model = RadynversionNet(inRepr, outRepr, dropout=dropout, zeroPadding=0, minSize=ndim_tot, numInvLayers=numInvLayers)

    # Construct the class that trains the model, the initial weighting between the losses, learning rate, and the initial number of epochs to train for.

    # load previous model if asked
    if load_model: model.load_state_dict(torch.load('models/gpu5_model.pt')) #% run_label))

    if do_double_nn:
        trainer = DoubleNetTrainer(model_f, model_r, data, dev, load_model=load_model)
        trainer.training_params(tot_epoch, lr=lr, fadeIn=fadeIn,
                                loss_latent=Loss.mmd_multiscale_on(dev, alphas=latentAlphas),
                                loss_fit=Loss.mse,ndata=ndata,sigma=sigma,seed=seed,batchSize=batchsize,usepars=usepars)
    else:
        trainer = RadynversionTrainer(model, data, dev, load_model=load_model)
        trainer.training_params(tot_epoch, lr=lr, fadeIn=fadeIn, zerosNoiseScale=zerosNoiseScale, wPred=wPred, wLatent=wLatent, wRev=wRev,
                                loss_latent=Loss.mmd_multiscale_on(dev, alphas=latentAlphas),
                                loss_backward=Loss.mmd_multiscale_on(dev, alphas=backwardAlphas),
                                loss_fit=Loss.mse,ndata=ndata,sigma=sigma,seed=seed,n_neurons=n_neurons,batchSize=batchsize,usepars=usepars,
                                y_noise_scale=y_noise_scale)
    totalEpochs = 0

    # Train the model for these first epochs with a nice graph that updates during training.

    losses = []
    wRevScale_tot = []
    beta_score_hist=[]
    beta_score_loop_hist=[]
    lossVec = [[] for _ in range(4)]
    lossLabels = ['L2 Line', 'MMD Latent', 'MMD Reverse', 'L2 Reverse']
    out = None
    alphaRange, mmdF, mmdB, idxF, idxB = [1,1], [1,1], [1,1], 0, 0

    try:
        tStart = time()
        olvec = np.zeros((r,r,int(trainer.numEpochs/plot_cadence)))
        adksVec = np.zeros((r,r,ndim_x,4,int(trainer.numEpochs/plot_cadence)))
        s = 0

        for epoch in range(trainer.numEpochs):
            print('Epoch %s/%s' % (str(epoch),str(trainer.numEpochs)))
            totalEpochs += 1

            if do_double_nn:
                trainer.scheduler_f.step()
        
                loss, indLosses = trainer.train(epoch,gen_inf_temp=gen_inf_temp,extra_z=extra_z, do_cnn=do_cnn)
            else:
                trainer.scheduler.step()

                loss, indLosses = trainer.train(epoch,gen_inf_temp=gen_inf_temp,extra_z=extra_z,do_covar=do_covar)

            if do_double_nn:
                # save trained model
                torch.save(model_f.state_dict(), 'models/%s_model_f.pt' % run_label)
                torch.save(model_r.state_dict(), 'models/%s_model_r.pt' % run_label)
            else:
                # save trained model
                torch.save(model.state_dict(), 'models/%s_model.pt' % run_label)

            # loop over a few cases and plot results in a grid
            if np.remainder(epoch,plot_cadence)==0:
                for k in range(ndim_x):
                    parname1 = parnames[k]
                    for nextk in range(ndim_x):
                        parname2 = parnames[nextk]
                        if nextk>k:
                            cnt = 0

                            # initialize plot for showing testing results
                            fig, axes = plt.subplots(r,r,figsize=(6,6),sharex='col',sharey='row')

                            for i in range(r):
                                for j in range(r):

                                    # convert data into correct format
                                    y_samps = np.tile(np.array(labels_test[cnt,:]),N_samp).reshape(N_samp,ndim_y)
                                    y_samps = torch.tensor(y_samps, dtype=torch.float)

                                    # add noise to y data (why?)
                                    #y = y_samps + y_noise_scale * torch.randn(N_samp, ndim_y, device=dev)
                                    if do_double_nn:
                                        y_samps = torch.cat([torch.randn(N_samp, ndim_z),
                                            y_samps], dim=1)
                                    else:
                                        y_samps = torch.cat([torch.randn(N_samp, ndim_z), zerosNoiseScale * 
                                            torch.zeros(N_samp, ndim_tot - ndim_y - ndim_z),
                                            y_samps], dim=1)
                                    y_samps = y_samps.to(dev)

                                    # use the network to predict parameters
                                    if do_double_nn:
                                        if do_cnn: 
                                            y_samps = y_samps.reshape(y_samps.shape[0],1,y_samps.shape[1]) 
                                            rev_x = model_r(y_samps)
                                        else:
                                            rev_x = model_r(y_samps)
                                    else: rev_x = model(y_samps, rev=True)
                                    rev_x = rev_x.cpu().data.numpy()

                                    # compute the n-d overlap
                                    if k==0 and nextk==1:
                                        ol = data_maker.overlap(samples[cnt,:,:ndim_x],rev_x[:,:ndim_x])
                                        olvec[i,j,s] = ol                                     

                                        # print A-D and K-S test
                                        ks_mcmc_arr, ks_inn_arr, ad_mcmc_arr, ad_inn_arr = result_stat_tests(rev_x[:, usepars], samples[cnt,:,:ndim_x], cnt, parnames)
                                        for p in usepars:
                                            for c in range(4):
                                                adksVec[i,j,p,c,s] = np.array([ks_mcmc_arr,ks_inn_arr,ad_mcmc_arr,ad_inn_arr])[c,p] 
                      
                                    # plot the samples and the true contours
                                    axes[i,j].clear()
                                    if latent==True:
                                        colors = z.cpu().detach().numpy()
                                        colors = np.linalg.norm(colors,axis=1)
                                        axes[i,j].scatter(rev_x[:,k], rev_x[:,nextk],c=colors,s=1.0,cmap='hsv',alpha=0.75, label='INN')
                                    else:
                                        axes[i,j].scatter(samples[cnt,:,k], samples[cnt,:,nextk],c='b',s=0.2,alpha=0.5, label='MCMC')
                                        axes[i,j].scatter(rev_x[:,k], rev_x[:,nextk],c='r',s=0.2,alpha=0.5, label='INN')
                                        axes[i,j].set_xlim([0,1])
                                        axes[i,j].set_ylim([0,1])                                   
                                    axes[i,j].plot(pos_test[cnt,k],pos_test[cnt,nextk],'+c',markersize=8, label='Truth')
                                    oltxt = '%.2f' % olvec[i,j,s]
                                    axes[i,j].text(0.90, 0.95, oltxt,
                                        horizontalalignment='right',
                                        verticalalignment='top',
                                            transform=axes[i,j].transAxes)
                                    matplotlib.rc('xtick', labelsize=8)     
                                    matplotlib.rc('ytick', labelsize=8) 
                                    axes[i,j].set_xlabel(parname1) if i==r-1 else axes[i,j].set_xlabel('')
                                    axes[i,j].set_ylabel(parname2) if j==0 else axes[i,j].set_ylabel('')
                                    if i == 0 and j == 0: axes[i,j].legend(loc='upper left', fontsize='x-small')
                                    cnt += 1

                            # save the results to file
                            fig.canvas.draw()
                            if latent==True:
                                plt.savefig('%s/latent_map_%d%d_%04d.png' % (out_dir,k,nextk,epoch),dpi=360)
                                plt.savefig('%s/latest/latent_map_%d%d.png' % (out_dir,k,nextk),dpi=360)
                            else:
                                plt.savefig('%s/posteriors_%d%d_%04d.png' % (out_dir,k,nextk,epoch),dpi=360)
                                plt.savefig('%s/latest/posteriors_%d%d.png' % (out_dir,k,nextk),dpi=360)
                            plt.close(fig)
                s += 1

            # plot overlap results
            if np.remainder(epoch,plot_cadence)==0:                
                fig, axes = plt.subplots(1,figsize=(6,6))
                for i in range(r):
                    for j in range(r):
                        color = next(axes._get_lines.prop_cycler)['color']
                        axes.semilogx(np.arange(epoch, step=plot_cadence),olvec[i,j,:int((epoch)/plot_cadence)],alpha=0.5, color=color)
                        axes.plot([int(epoch)],[olvec[i,j,int(epoch/plot_cadence)]],'.', color=color)
                axes.grid()
                axes.set_ylabel('overlap')
                axes.set_xlabel('epoch')
                axes.set_ylim([0,1])
                plt.savefig('%s/latest/overlap_logscale.png' % out_dir, dpi=360)
                plt.close(fig)      

                fig, axes = plt.subplots(1,figsize=(6,6))
                for i in range(r):
                    for j in range(r):
                        color = next(axes._get_lines.prop_cycler)['color']
                        axes.plot(np.arange(epoch, step=plot_cadence),olvec[i,j,:int((epoch)/plot_cadence)],alpha=0.5, color=color)
                        axes.plot([int(epoch)],[olvec[i,j,int(epoch/plot_cadence)]],'.', color=color)
                axes.grid()
                axes.set_ylabel('overlap')
                axes.set_xlabel('epoch')
                axes.set_ylim([0,1])
                plt.savefig('%s/latest/overlap.png' % out_dir, dpi=360)
                plt.close(fig)

                if do_double_nn:
                    # plot predicted time series vs. actually time series examples
                    model=None
                    plot_y_test(model,N_samp,usepars,sigma,ndim_x,ndim_y,ndim_z,ndim_tot,out_dir,r,epoch,conv=False,model_f=model_f,model_r=model_r,do_double_nn=do_double_nn,do_cnn=do_cnn)

                    # make y_dist_plot
                    plot_y_dist(model,ndim_x,ndim_y,ndim_z,ndim_tot,usepars,sigma,out_dir,epoch,conv=False,model_f=model_f,model_r=model_r,do_double_nn=do_double_nn,do_cnn=do_cnn)

                    plot_x_evolution(model,ndim_x,ndim_y,ndim_z,ndim_tot,sigma,parnames,out_dir,epoch,conv=False,model_f=model_f,model_r=model_r,do_double_nn=do_double_nn,do_cnn=do_cnn)

                    plot_z_dist(model,ndim_x,ndim_y,ndim_z,ndim_tot,usepars,sigma,out_dir,epoch,conv=False,model_f=model_f,model_r=model_r,do_double_nn=do_double_nn,do_cnn=do_cnn)

                else:
                    # plot predicted time series vs. actually time series examples
                    plot_y_test(model,N_samp,usepars,sigma,ndim_x,ndim_y,ndim_z,ndim_tot,out_dir,r,epoch,conv=False)

                    # make y_dist_plot
                    plot_y_dist(model,ndim_x,ndim_y,ndim_z,ndim_tot,usepars,sigma,out_dir,epoch,conv=False)

                    plot_x_evolution(model,ndim_x,ndim_y,ndim_z,ndim_tot,sigma,parnames,out_dir,epoch,conv=False)

                    plot_z_dist(model,ndim_x,ndim_y,ndim_z,ndim_tot,usepars,sigma,out_dir,epoch,conv=False)

                # plot evolution of y
                if not do_double_nn:
                    for c in range(ndim_x):
                        plot_y_evolution(model,c,parnames,ndim_x,ndim_y,ndim_z,ndim_tot,out_dir,epoch,conv=False,model_f=model_f,model_r=model_r,do_double_nn=True,do_cnn=do_cnn) 

                # plot ad and ks results [ks_mcmc_arr,ks_inn_arr,ad_mcmc_arr,ad_inn_arr]
                for p in range(ndim_x):
                    fig_ks, axis_ks = plt.subplots(1,figsize=(6,6)) 
                    fig_ad, axis_ad = plt.subplots(1,figsize=(6,6))
                    for i in range(r):
                        for j in range(r):
                            color_ks = next(axis_ks._get_lines.prop_cycler)['color'] 
                            axis_ks.semilogx(np.arange(tot_epoch, step=plot_cadence),adksVec[i,j,p,0,:],'--',alpha=0.5,color=color_ks)
                            axis_ks.semilogx(np.arange(tot_epoch, step=plot_cadence),adksVec[i,j,p,1,:],alpha=0.5,color=color_ks)
                            axis_ks.plot([int(epoch)],[adksVec[i,j,p,1,int(epoch/plot_cadence)]],'.', color=color_ks) 
                            axis_ks.set_yscale('log')

                            color_ad = next(axis_ad._get_lines.prop_cycler)['color']
                            axis_ad.semilogx(np.arange(tot_epoch, step=plot_cadence),adksVec[i,j,p,2,:],'--',alpha=0.5,color=color_ad)
                            axis_ad.semilogx(np.arange(tot_epoch, step=plot_cadence),adksVec[i,j,p,3,:],alpha=0.5,color=color_ad)
                            axis_ad.plot([int(epoch)],[adksVec[i,j,p,3,int(epoch/plot_cadence)]],'.',color=color_ad)
                            axis_ad.set_yscale('log')

                    axis_ks.set_xlabel('Epoch')
                    axis_ad.set_xlabel('Epoch')
                    axis_ks.set_ylabel('KS Statistic')
                    axis_ad.set_ylabel('AD Statistic')
                    fig_ks.savefig('%s/latest/ks_%s_stat.png' % (out_dir,parnames[p]), dpi=360)
                    fig_ad.savefig('%s/latest/ad_%s_stat.png' % (out_dir,parnames[p]), dpi=360)
                    plt.close(fig_ks)
                    plt.close(fig_ad)

            if ((epoch % 10 == 0) & (epoch>5)):

                #fig, axis = plt.subplots(4,1, figsize=(10,8))
                #fig.canvas.draw()
                #axis[0].clear()
                #axis[1].clear()
                #axis[2].clear()
                #axis[3].clear()
                for i in range(len(indLosses)):
                    lossVec[i].append(indLosses[i])
                losses.append(loss)
                #fig.suptitle('Current Loss: %.2e, min loss: %.2e' % (loss, np.nanmin(np.abs(losses))))
                #axis[0].semilogy(np.arange(len(losses)), np.abs(losses))
                #for i, lo in enumerate(lossVec):
                #    axis[1].semilogy(np.arange(len(losses)), lo, '--', label=lossLabels[i])
                #axis[1].legend(loc='upper left')
                #tNow = time()
                #elapsed = int(tNow - tStart)
                #eta = int((tNow - tStart) / (epoch + 1) * trainer.numEpochs) - elapsed

                #if epoch % 2 == 0:
                #    mses = trainer.test(samples,maxBatches=1,extra_z=extra_z)
                #    lineProfiles = mses[2]
                
                if epoch % 10 == 0 and review_mmd and epoch >=600:
                    print('Reviewing alphas')
                    alphaRange, mmdF, mmdB, idxF, idxB = trainer.review_mmd()
                
                #axis[3].semilogx(alphaRange, mmdF, label='Latent Space')
                #axis[3].semilogx(alphaRange, mmdB, label='Backward')
                #axis[3].semilogx(alphaRange[idxF], mmdF[idxF], 'ro')
                #axis[3].semilogx(alphaRange[idxB], mmdB[idxB], 'ro')
                #axis[3].legend()

                #testTime = time() - tNow
                #axis[2].plot(lineProfiles[0, model.outSchema.timeseries].cpu().numpy())
                #for a in axis:
                #    a.grid()
                #axis[3].set_xlabel('Epochs: %d, Elapsed: %d s, ETA: %d s (Testing: %d s)' % (epoch, elapsed, eta, testTime))
            
                
                #fig.canvas.draw()
                #fig.savefig('%slosses-wave-tot.pdf' % out_dir)
                #plt.close(fig)

                # make latent space plots
                """
                if epoch % plot_cadence == 0:
                    labels_z = []
                    for lab_idx in range(ndim_z):
                        labels_z.append(r"latent%d" % lab_idx)
                    fig_latent = corner.corner(lineProfiles[:, model.outSchema.LatentSpace].cpu().numpy(), 
                                               plot_contours=False, labels=labels_z)
                    fig_latent.savefig('%s/latest/latent_space.pdf' % out_dir)
                    print('Plotted latent space')
                    plt.close(fig_latent)
                """

                # make non-logscale loss plot
                fig_loss, axes_loss = plt.subplots(1,figsize=(10,8))
                wRevScale_tot.append(trainer.wRevScale)
                axes_loss.grid()
                axes_loss.set_ylabel('Loss')
                axes_loss.set_xlabel('Epochs elapsed: %s' % epoch)
                axes_loss.semilogy(np.arange(len(losses)), np.abs(losses), label='Total')
                for i, lo in enumerate(lossVec):
                    axes_loss.semilogy(np.arange(len(losses)), lo, label=lossLabels[i])
                axes_loss.semilogy(np.arange(len(losses)), wRevScale_tot, label='fadeIn')
                axes_loss.legend(loc='upper left')
                plt.savefig('%s/latest/losses.png' % out_dir)
                plt.close(fig)

                # make log scale loss plot
                fig_loss, axes_loss = plt.subplots(1,figsize=(10,8))
                axes_loss.grid()
                axes_loss.set_ylabel('Loss')
                axes_loss.set_xlabel('Epochs elapsed: %s' % epoch)
                axes_loss.plot(np.arange(len(losses)), np.abs(losses), label='Total')
                for i, lo in enumerate(lossVec):
                    axes_loss.plot(np.arange(len(losses)), lo, label=lossLabels[i])
                axes_loss.plot(np.arange(len(losses)), wRevScale_tot, label='fadeIn')
                axes_loss.set_xscale('log')
                axes_loss.set_yscale('log')
                axes_loss.legend(loc='upper left')
                plt.savefig('%s/latest/losses_logscale.png' % out_dir)
                plt.close(fig)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nTraining took {(time()-tStart)/60:.2f} minutes\n")
main()
exit()
