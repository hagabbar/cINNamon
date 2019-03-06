
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

from Inn2 import RadynversionNet, AtmosData, RadynversionTrainer
import loss as Loss
import pickle
import numpy as np
import scipy
from scipy.stats import multivariate_normal as mvn
from scipy.special import logit, expit
from scipy.stats import uniform, gaussian_kde, ks_2samp, anderson_ksamp
from scipy import stats
from scipy.signal import butter, lfilter, freqs, resample
from scipy.interpolate import interp1d
#import matplotlib
#matplotlib.use('Agg')
import matplotlib
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.utils.data
import h5py
import os
from sys import exit

from time import time

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import rev_multiplicative_layer, F_fully_connected, F_conv

import chris_data as data_maker
import bilby_pe

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:95% !important; }</style>"))

# global parameters
sig_model = 'sg'   # the signal model to use
usepars = [0,1,2,3]    # parameter indices to use
run_label='gpu0'
out_dir = "/home/hunter.gabbard/public_html/CBC/cINNamon/gw_results/multipar/%s/" % run_label
do_posterior_plots=True
ndata=1024           # length of 1 data sample
ndim_x=len(usepars)           # number of parameters to PE on
ndim_y = ndata     # length of 1 data sample

ndim_z = 6     # size of latent space

Ngrid=20
n_neurons = 0
ndim_tot = max(ndim_x,ndim_y+ndim_z) + n_neurons # 384     
r = 3              # the grid dimension for the output tests
sigma = 1.0        # the noise std
seed = 1           # seed for generating data
test_split = r*r   # number of testing samples to use

N_samp = 1000 # number of test samples to use after training
plot_cadence = 50 # make plots every N iterations
numInvLayers=5
dropout=0.0
batchsize=1600
filtsize = 3       # TODO
clamp=2.0          # TODO
tot_dataset_size=1000 # 2**20 TODO really should use 1e8 once cpu is fixed

tot_epoch=50000
lr=1.0e-3
zerosNoiseScale=5e-2
y_noise_scale = 5e-2

wPred=4000.0        #4000.0
wLatent= 900.0     #900.0
wRev= 1000.0        #1000.0

template_dir = 'gw_data/templates/'
event_name = 'gw150914'
tag = '_srate-1024hz_oversamp_python3'
add_noise_real=False

latentAlphas=None
backwardAlphas=None # [1.4, 2, 5.5, 7]
conv_nn = False    # Choose to use convolutional layers. TODO
multi_par=True
load_dataset=False
do_contours=True
do_mcmc=True
dataLocation1 = 'benchmark_data_%s.h5py' % run_label
T = 1.0           # length of time series (s)
dt = T/ndata        # sampling time (Sec)
fnyq = 0.5/dt   # Nyquist frequency (Hz)
if multi_par==True: bound = [0.0,1.0,0.0,1.0,0.0,1.0*fnyq,0.0,3.0,0.0,1.0]
else: bound = [0.0,1.0,0.0,1.0] # effective bound for the liklihood
parnames=['mc','eta','phi','t0']  # eventually need to change q for eta

def make_contour_plot(ax,x,y,dataset,color='red',flip=False, kernel_lalinf=False, kernel_cnn=False, bounds=[0.0,1.0,0.0,1.0]):
    """ Module used to make contour plots in pe scatter plots.
    Parameters
    ----------
    ax: matplotlib figure
        a matplotlib figure instance
    x: 1D numpy array
        pe sample parameters for x-axis
    y: 1D numpy array
        pe sample parameters for y-axis
    dataset: 2D numpy array
        array containing both parameter estimates
    color:
        color of contours in plot
    flip:
        if True: transpose parameter estimates array. if False: do not transpose parameter estimates
        TODO: This is not used, so should remove
    Returns
    -------
    kernel: scipy kernel
        gaussian kde of the input dataset
    """
    # Make a 2d normed histogram
    H,xedges,yedges=np.histogram2d(x,y,bins=10,normed=True)

    if flip == True:
        H,xedges,yedges=np.histogram2d(y,x,bins=10,normed=True)
        dataset = np.array([dataset[1,:],dataset[0,:]])

    norm=H.sum() # Find the norm of the sum
    # Set contour levels
    contour1=0.99
    contour2=0.90
    contour3=0.68

    # Set target levels as percentage of norm
    target1 = norm*contour1
    target2 = norm*contour2
    target3 = norm*contour3

    # Take histogram bin membership as proportional to Likelihood
    # This is true when data comes from a Markovian process
    def objective(limit, target):
        w = np.where(H>limit)
        count = H[w]
        return count.sum() - target

    # Find levels by summing histogram to objective
    level1= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target1,))
    level2= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target2,))
    level3= scipy.optimize.bisect(objective, H.min(), H.max(), args=(target3,))

    # For nice contour shading with seaborn, define top level
    #level4=H.max()
    levels=[level1,level2,level3]

    # Pass levels to normed kde plot
    #sns.kdeplot(x,y,shade=True,ax=ax,n_levels=levels,cmap=color,alpha=0.5,normed=True)
    #X, Y = np.mgrid[np.min(x):np.max(x):100j, np.min(y):np.max(y):100j]
    X, Y = np.mgrid[bounds[0]:bounds[1]:100j, bounds[2]:bounds[3]:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    if not kernel_lalinf or not kernel_cnn: kernel = gaussian_kde(dataset)
    Z = np.reshape(kernel(positions).T, X.shape)
    ax.contour(X,Y,Z,levels=levels,alpha=0.5,colors=color,linewidths=0.6)
    #ax.set_aspect('equal')

    return kernel

def load_gw_data(seed=0):
    # load first time series / pars template pickle file
    np.random.seed(seed)

    file_idx_list = []
    h5py_ts = h5py.File("%s%s_ts_0_%sSamp%s.hdf5" % (template_dir,event_name,tot_dataset_size,tag),"r")
    ts = h5py_ts['time_series/ts'][:]
    h5py_par = h5py.File("%s%s_params_0_%sSamp%s.hdf5" % (template_dir,event_name,tot_dataset_size,tag),"r")
    par = h5py_par['parameters/par'][:]
    if len(file_idx_list) > 0:
        ts = np.array(ts[:-1])
        par = np.array(par[:-1])
    else:
        ts = np.array(ts)
        par = np.array(par)

    par = np.reshape(par,(par.shape[0],2))
    print("loading file: _ts_0_%sSamp.hdf5" % (tot_dataset_size))
    print("loading file: _params_0_%sSamp.hdf5" % (tot_dataset_size))

    # iterate over all other data files and load them
    for idx in file_idx_list:
        h5py_ts = open("%s_ts_%s_%sSamp%s.hdf5" % (template_dir,str(idx),tot_dataset_size,tag),"rb")
        ts_new = h5py_ts['time_series/ts'][:]
        ts = np.vstack((ts,ts_new))

        # load corresponding parameters template pickle file
        h5py_par = open("%s_params_%s_%sSamp%s.hdf5" % (template_dir,str(idx),tot_dataset_size,tag),"rb")
        par_new = np.array(h5py_par['parameters/par'][:])
        par_new = np.reshape(par_new,(par_new.shape[0],1))
        par = np.vstack((par,par_new))

        print("loading file: _ts_%s_%sSamp.hdf5" % (str(idx),tot_dataset_size))
        print("loading file: _params_%s_%sSamp.hdf5" % (str(idx),tot_dataset_size))

        if idx < file_idx_list[-1]:
            ts = ts[:-1]
            par = par[:-1]

    ts = [ts]

    signal_ts_noisefree = np.reshape(ts[0], (ts[0].shape[0],ts[0].shape[2]))

    signal_pars = par

    # shuffle training set
    shuffling = np.random.permutation(tot_dataset_size)
    signal_ts_noisefree = signal_ts_noisefree[shuffling]
    signal_pars = signal_pars[shuffling]
    names = [parnames[int(i)] for i in usepars]
    xvec = np.arange(ndata)/float(ndata)

    signal_ts_noise = np.zeros(signal_ts_noisefree.shape)
    for i in range(len(signal_ts_noisefree)):
        signal_ts_noise[i] = signal_ts_noisefree[i] + np.random.normal(loc=0.0, scale=sigma, size=ndata)

    return signal_ts_noise, signal_pars, xvec, signal_ts_noisefree, names 

def main():

    # setup output directory - if it does not exist
    os.system('mkdir -p %s' % out_dir)

    if not load_dataset:
        signal_train_images, sig, signal_train_pars = bilby_pe.run(N_gen=tot_dataset_size,make_train_samp=True,make_test_samp=False)

        #signal_train_images = signal_train_images.reshape((signal_train_images.shape[0],signal_train_images.shape[2]))
        # declare gw variants of positions and labels
        labels = torch.tensor(signal_train_images, dtype=torch.float)
        pos = torch.tensor(signal_train_pars, dtype=torch.float)
        sig = torch.tensor(sig, dtype=torch.float)
        x = np.arange(ndata)/float(ndata)
        
        print("Loaded data ...")

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
    print('Computing Bilby posterior samples')
    exit()
    if do_mcmc or not load_dataset:
        # precompute true posterior samples on the test data
        cnt = 0
        samples = np.zeros((r*r,N_samp,ndim_x))
        for i in range(r):
            for j in range(r):
                samples[cnt,:,:] = data_maker.get_lik(np.array(labels_test[cnt,:]).flatten(),np.array(pos_test[cnt,:]),
                                                      out_dir,cnt,sigma=sigma,usepars=usepars,Nsamp=N_samp)
                print(samples[cnt,:10,:])
                cnt += 1

        # save computationaly expensive mcmc/waveform runs
        if load_dataset==True:
            hf = h5py.File('benchmark_data_%s.h5py' % run_label, 'w')
            hf.create_dataset('pos', data=data.pos)
            hf.create_dataset('labels', data=data.labels)
            hf.create_dataset('x', data=data.x)
            hf.create_dataset('sig', data=data.sig)
            hf.create_dataset('parnames', data=parnames)
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
            axes[i,j].plot(data.x,np.array(labels_test[cnt,:]),'.')
            axes[i,j].plot(data.x,np.array(sig_test[cnt,:]),'-')
            cnt += 1
            axes[i,j].axis([0,1,-1.5,1.5])
            axes[i,j].set_xlabel('time') if i==r-1 else axes[i,j].set_xlabel('')
            axes[i,j].set_ylabel('h(t)') if j==0 else axes[i,j].set_ylabel('')
    plt.savefig('%stest_distribution.png' % out_dir,dpi=360)
    plt.close()

    exit()
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
                         axes[i,j].scatter(samples[cnt,:,k], samples[cnt,:,nextk],c='b',s=0.5,alpha=0.5)
                         axes[i,j].plot(pos_test[cnt,k],pos_test[cnt,nextk],'+c',markersize=8)
                         axes[i,j].set_xlim([0,1])
                         axes[i,j].set_ylim([0,1])
                         axes[i,j].set_xlabel(parname1) if i==r-1 else axes[i,j].set_xlabel('')
                         axes[i,j].set_ylabel(parname2) if j==0 else axes[i,j].set_ylabel('')
                         
                         cnt += 1

                # save the results to file
                fig.canvas.draw()
                plt.savefig('%strue_samples_%d%d.png' % (out_dir,k,nextk),dpi=360)

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

    inRepr = [('amp', 1), ('t0', 1), ('tau', 1), ('phi', 1), ('!!PAD',)]
    outRepr = [('LatentSpace', ndim_z), ('!!PAD',), ('timeseries', data.atmosOut.shape[1])]
    model = RadynversionNet(inRepr, outRepr, dropout=dropout, zeroPadding=0, minSize=ndim_tot, numInvLayers=numInvLayers)

    # Construct the class that trains the model, the initial weighting between the losses, learning rate, and the initial number of epochs to train for.


    trainer = RadynversionTrainer(model, data, dev)
    trainer.training_params(tot_epoch, lr=lr, zerosNoiseScale=zerosNoiseScale, wPred=wPred, wLatent=wLatent, wRev=wRev,
                            loss_latent=Loss.mmd_multiscale_on(dev, alphas=latentAlphas),
                            loss_backward=Loss.mmd_multiscale_on(dev, alphas=backwardAlphas),
                            loss_fit=Loss.mse)
    totalEpochs = 0

    # Train the model for these first epochs with a nice graph that updates during training.


    losses = []
    beta_score_hist=[]
    beta_score_loop_hist=[]
    lossVec = [[] for _ in range(4)]
    lossLabels = ['L2 Line', 'MMD Latent', 'MMD Reverse', 'L2 Reverse']
    out = None
    alphaRange, mmdF, mmdB, idxF, idxB = [1,1], [1,1], [1,1], 0, 0

    try:
        tStart = time()
        olvec = np.zeros((r,r,int(tot_epoch/plot_cadence)))
        s = 0

        for epoch in range(trainer.numEpochs):
            print('Epoch %s/%s' % (str(epoch),str(trainer.numEpochs)))
            totalEpochs += 1

            trainer.scheduler.step()
        
            loss, indLosses = trainer.train(epoch)

            # loop over a few cases and plot results in a grid
            if np.remainder(epoch,plot_cadence)==0:
                for k in range(ndim_x):
                    parname1 = parnames[k]
                    for nextk in range(ndim_x):
                        parname2 = parnames[nextk]
                        if nextk>k:
                            cnt = 0

                            # initialize 2D plots for showing testing results
                            fig, axes = plt.subplots(r,r,figsize=(6,6),sharex='col',sharey='row')

                            # initialize 1D plots for showing testing results
                            fig_1d, axes_1d = plt.subplots(r,r,figsize=(6,6),sharex='col',sharey='row')

                            for i in range(r):
                                for j in range(r):

                                    # convert data into correct format
                                    y_samps = np.tile(np.array(labels_test[cnt,:]),N_samp).reshape(N_samp,ndim_y)
                                    y_samps = torch.tensor(y_samps, dtype=torch.float)
                                    y_samps += y_noise_scale * torch.randn(N_samp, ndim_y)
                                    y_samps = torch.cat([torch.randn(N_samp, ndim_z), zerosNoiseScale * 
                                        torch.zeros(N_samp, ndim_tot - ndim_y - ndim_z),
                                        y_samps], dim=1)
                                    y_samps = y_samps.to(dev)

                                    # use the network to predict parameters
                                    rev_x = model(y_samps, rev=True)
                                    rev_x = rev_x.cpu().data.numpy()

                                    # compute the n-d overlap
                                    if k==0 and nextk==1:
                                        ol = data_maker.overlap(samples[cnt,:,:ndim_x],rev_x[:,:ndim_x])
                                        olvec[i,j,s] = ol                                     

                                    def confidence_bd(samp_array):
                                        """
                                        compute confidence bounds for a given array
                                        """
                                        cf_bd_sum_lidx = 0
                                        cf_bd_sum_ridx = 0
                                        cf_bd_sum_left = 0
                                        cf_bd_sum_right = 0
                                        cf_perc = 0.05

                                        cf_bd_sum_lidx = np.sort(samp_array)[int(len(samp_array)*cf_perc)]
                                        cf_bd_sum_ridx = np.sort(samp_array)[int(len(samp_array)*(1.0-cf_perc))]

                                        return [cf_bd_sum_lidx, cf_bd_sum_ridx]

                                    # plot the 2D samples and the true contours
                                    true_cfbd_x = confidence_bd(samples[cnt,:,k])
                                    true_cfbd_y = confidence_bd(samples[cnt,:,nextk]) 
                                    pred_cfbd_x = confidence_bd(rev_x[:,k])
                                    pred_cfbd_y = confidence_bd(rev_x[:,nextk])
                                    axes[i,j].clear()
                                    axes[i,j].scatter(samples[cnt,:,k], samples[cnt,:,nextk],c='b',s=0.2,alpha=0.5)
                                    axes[i,j].scatter(rev_x[:,k], rev_x[:,nextk],c='r',s=0.2,alpha=0.5)
                                    axes[i,j].plot(pos_test[cnt,k],pos_test[cnt,nextk],'+c',markersize=8)
                                    #axes[i,j].axvline(x=true_cfbd_x[0], linewidth=0.5, color='b')
                                    #axes[i,j].axvline(x=true_cfbd_x[1], linewidth=0.5, color='b')
                                    #axes[i,j].axhline(y=true_cfbd_y[0], linewidth=0.5, color='b')
                                    #axes[i,j].axhline(y=true_cfbd_y[1], linewidth=0.5, color='b')
                                    #axes[i,j].axvline(x=pred_cfbd_x[0], linewidth=0.5, color='r')
                                    #axes[i,j].axvline(x=pred_cfbd_x[1], linewidth=0.5, color='r')
                                    #axes[i,j].axhline(y=pred_cfbd_y[0], linewidth=0.5, color='r')
                                    #axes[i,j].axhline(y=pred_cfbd_y[1], linewidth=0.5, color='r')
                                    axes[i,j].set_xlim([0,1])
                                    axes[i,j].set_ylim([0,1])
                                    oltxt = '%.2f' % olvec[i,j,s]
                                    axes[i,j].text(0.90, 0.95, oltxt,
                                        horizontalalignment='right',
                                        verticalalignment='top',
                                            transform=axes[i,j].transAxes)
                                    matplotlib.rc('xtick', labelsize=8)     
                                    matplotlib.rc('ytick', labelsize=8) 
                                    axes[i,j].set_xlabel(parname1) if i==r-1 else axes[i,j].set_xlabel('')
                                    axes[i,j].set_ylabel(parname2) if j==0 else axes[i,j].set_ylabel('')
                                   
                                    # plot the 1D samples and the 5% confidence bounds
                                    axes_1d[i,j].clear()
                                    axes_1d[i,j].hist(samples[cnt,:,k],color='b',bins=100,alpha=0.5)
                                    axes_1d[i,j].hist(rev_x[:,k],color='r',bins=100,alpha=0.5)
                                    axes_1d[i,j].axvline(x=pos_test[cnt,k], linewidth=0.5, color='black')
                                    axes_1d[i,j].axvline(x=confidence_bd(samples[cnt,:,k])[0], linewidth=0.5, color='b')
                                    axes_1d[i,j].axvline(x=confidence_bd(samples[cnt,:,k])[1], linewidth=0.5, color='b')
                                    axes_1d[i,j].axvline(x=confidence_bd(rev_x[:,k])[0], linewidth=0.5, color='r')
                                    axes_1d[i,j].axvline(x=confidence_bd(rev_x[:,k])[1], linewidth=0.5, color='r')
                                    axes_1d[i,j].set_xlim([0,1])
                                    axes_1d[i,j].text(0.90, 0.95, oltxt,
                                        horizontalalignment='right',
                                        verticalalignment='top',
                                            transform=axes_1d[i,j].transAxes)
                                    axes_1d[i,j].set_xlabel(parname1) if i==r-1 else axes_1d[i,j].set_xlabel('')

                                    cnt += 1
                            # save the results to file
                            fig_1d.canvas.draw()
                            fig_1d.savefig('%sposteriors-1d_%d_%04d.png' % (out_dir,k,epoch),dpi=360)
                            fig_1d.savefig('%slatest-1d_%d.png' % (out_dir,k),dpi=360)
                            #fig_1d.close()

                            fig.canvas.draw()
                            fig.savefig('%sposteriors-2d_%d%d_%04d.png' % (out_dir,k,nextk,epoch),dpi=360)
                            fig.savefig('%slatest-2d_%d%d.png' % (out_dir,k,nextk),dpi=360)
                            #fig.close()
                s += 1

            # plot overlap results
            if np.remainder(epoch,plot_cadence)==0:
                fig_log = plt.figure(figsize=(6,6))             
                axes_log = fig_log.add_subplot(1,1,1)
                for i in range(r):
                    for j in range(r):
                        axes_log.semilogx(np.arange(tot_epoch,step=plot_cadence),olvec[i,j,:],alpha=0.5)
                axes_log.grid()
                axes_log.set_ylabel('overlap')
                axes_log.set_xlabel('epoch (log)')
                axes_log.set_ylim([0,1])
                plt.savefig('%soverlap_logscale.png' % out_dir, dpi=360)
                plt.close()      

                fig = plt.figure(figsize=(6,6))
                axes = fig.add_subplot(1,1,1)
                for i in range(r):
                    for j in range(r):
                        axes.plot(np.arange(tot_epoch,step=plot_cadence),olvec[i,j,:],alpha=0.5)
                axes.grid()
                axes.set_ylabel('overlap')
                axes.set_xlabel('epoch')
                axes.set_ylim([0,1])
                plt.savefig('%soverlap.png' % out_dir, dpi=360)
                plt.close()  

            #egg = True
            #if egg==False:
            if np.remainder(epoch,plot_cadence)==0 and (epoch>5):
                fig, axis = plt.subplots(4,1, figsize=(10,8))
                #fig.show()
                fig.canvas.draw()
                axis[0].clear()
                axis[1].clear()
                axis[2].clear()
                axis[3].clear()
                for i in range(len(indLosses)):
                    lossVec[i].append(indLosses[i])
                losses.append(loss)
                fig.suptitle('Current Loss: %.2e, min loss: %.2e' % (loss, np.nanmin(np.abs(losses))))
                axis[0].semilogy(np.arange(len(losses)), np.abs(losses))
                for i, lo in enumerate(lossVec):
                    axis[1].semilogy(np.arange(len(losses)), lo, '--', label=lossLabels[i])
                axis[1].legend(loc='upper left')
                tNow = time()
                elapsed = int(tNow - tStart)
                eta = int((tNow - tStart) / (epoch + 1) * trainer.numEpochs) - elapsed

                if epoch % 2 == 0:
                    mses = trainer.test(maxBatches=1)
                    lineProfiles = mses[2]
                
                if epoch % 10 == 0:
                    alphaRange, mmdF, mmdB, idxF, idxB = trainer.review_mmd()
                
                axis[3].semilogx(alphaRange, mmdF, label='Latent Space')
                axis[3].semilogx(alphaRange, mmdB, label='Backward')
                axis[3].semilogx(alphaRange[idxF], mmdF[idxF], 'ro')
                axis[3].semilogx(alphaRange[idxB], mmdB[idxB], 'ro')
                axis[3].legend()

                testTime = time() - tNow
                axis[2].plot(lineProfiles[0, model.outSchema.timeseries].cpu().numpy())
                for a in axis:
                    a.grid()
                axis[3].set_xlabel('Epochs: %d, Elapsed: %d s, ETA: %d s (Testing: %d s)' % (epoch, elapsed, eta, testTime))
            
                
                fig.canvas.draw()
                fig.savefig('%slosses.pdf' % out_dir)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nTraining took {(time()-tStart)/60:.2f} minutes\n")
main()
exit()
