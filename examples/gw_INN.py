import torch
import torch.optim

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import pickle, h5py

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import rev_multiplicative_layer, F_fully_connected

import data

from sys import exit
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cuda_dev = "7" # define GPU to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cuda_dev

# define some global parameters
event_name = 'gw150914'        # event name
template_dir = 'gw_data/templates/'    # location of training templates directory
data_dir = 'gw_data/data/'             # data folder location
tag = '_srate-1024hz_oversamp_python3' # special tag for some files used
sanity_check_file = 'gw150914_cnn_sanity_check_ts_mass-time-vary_srate-1024hz.sav' # name of file used for checking absolute best performance of CNN
total_temp_num = 50000         # total number of gw templates to load
n_sig = 1

# load in lalinference converted chirp mass and inverse mass ratio parameters
with open("%s%s_mc_q_lalinf_post_srate-1024_python3.sav" % (data_dir,event_name), 'rb') as f:
    pickle_lalinf_pars = pickle.load(f)

#pickle_lalinf_pars = open("%s%s_mc_q_lalinf_post.sav" % (data_dir,event_name), "rb")
lalinf_pars = pickle_lalinf_pars

# define output path
out_path = '/home/hunter.gabbard/public_html/CBC/gw_INN/%s' % event_name
# setup output directory - if it does not exist
os.system('mkdir -p %s' % out_path)

# define bbh parameters class
class bbhparams:
    def __init__(self,mc,M,eta,m1,m2,ra,dec,iota,phi,psi,idx,fmin,snr,SNR):
        self.mc = mc
        self.M = M
        self.eta = eta
        self.m1 = m1
        self.m2 = m2
        self.ra = ra
        self.dec = dec
        self.iota = iota
        self.phi = phi
        self.psi = psi
        self.idx = idx
        self.fmin = fmin
        self.snr = snr
        self.SNR = SNR

def MMD_multiscale(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    for a in [0.2, 0.5, 0.9, 1.3]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    return torch.mean(XX + YY - 2.*XY)

def fit(input, target):
    return torch.mean((input - target)**2)

def train(model,train_loader,n_its_per_epoch,zeros_noise_scale,batch_size,ndim_tot,ndim_x,ndim_y,ndim_z,y_noise_scale,optimizer,lambd_predict,loss_fit,lambd_latent,loss_latent,lambd_rev,loss_backward,i_epoch=0):
    model.train()

    l_tot = 0
    batch_idx = 0

    t_start = time()

    loss_factor = 600**(float(i_epoch) / 300) / 600
    if loss_factor > 1:
        loss_factor = 1

    for x, y in train_loader:
        batch_idx += 1
        if batch_idx > n_its_per_epoch:
            break

        x, y = x.to(device), y.to(device)

        y_clean = y.clone()
        # zeros_noise_scale *
        pad_x = torch.randn(batch_size, ndim_tot -
                                                ndim_x, device=device)
        # zeros_noise_scale *
        pad_yz = torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)

        y += torch.randn(batch_size, ndim_y, dtype=torch.float, device=device) #* y_noise_scale

        x, y = (torch.cat((x, pad_x),  dim=1),
                torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y),
                          dim=1))

        optimizer.zero_grad()

        # Forward step:

        output = model(x)

        # Shorten output, and remove gradients wrt y, for latent loss
        y_short = torch.cat((y[:, :ndim_z], y[:, -ndim_y:]), dim=1)

        l = lambd_predict * loss_fit(output[:, ndim_z:], y[:, ndim_z:])

        output_block_grad = torch.cat((output[:, :ndim_z],
                                       output[:, -ndim_y:].data), dim=1)

        l += lambd_latent * loss_latent(output_block_grad, y_short)
        l_tot += l.data.item()

        l.backward()

        # Backward step:
        #zeros_noise_scale *
        pad_yz = torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)
        y = y_clean + torch.randn(batch_size, ndim_y, device=device) #* y_noise_scale

        orig_z_perturbed = (output.data[:, :ndim_z] +
                            torch.randn(batch_size, ndim_z, device=device)) #* y_noise_scale))
        y_rev = torch.cat((orig_z_perturbed, pad_yz,
                           y), dim=1)
        y_rev_rand = torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz,
                                y), dim=1)

        output_rev = model(y_rev, rev=True)
        output_rev_rand = model(y_rev_rand, rev=True)

        l_rev = (
            lambd_rev
            * loss_factor
            * loss_backward(output_rev_rand[:, :ndim_x],
                            x[:, :ndim_x])
        )

        l_rev += 0.50 * lambd_predict * loss_fit(output_rev, x)

        l_tot += l_rev.data.item()
        l_rev.backward()

        for p in model.parameters():
            p.grad.data.clamp_(-15.00, 15.00)

        optimizer.step()

        #     print('%.1f\t%.5f' % (
        #                              float(batch_idx)/(time()-t_start),
        #                              l_tot / batch_idx,
        #                            ), flush=True)

    return l_tot / batch_idx

def load_gw_data():
    # load first time series / pars template pickle file

    file_idx_list = []
    h5py_ts = h5py.File("%s%s_ts_0_%sSamp%s.hdf5" % (template_dir,event_name,total_temp_num,tag),"r")
    ts = h5py_ts['time_series/ts'][:]
    h5py_par = h5py.File("%s%s_params_0_%sSamp%s.hdf5" % (template_dir,event_name,total_temp_num,tag),"r")
    par = h5py_par['parameters/par'][:]
    if len(file_idx_list) > 0:
        ts = np.array(ts[:-1])
        par = np.array(par[:-1])
    else:
        ts = np.array(ts)
        par = np.array(par)

    par = np.reshape(par,(par.shape[0],2))
    print("loading file: _ts_0_%sSamp.hdf5" % (total_temp_num))
    print("loading file: _params_0_%sSamp.hdf5" % (total_temp_num))

    # iterate over all other data files and load them
    for idx in file_idx_list:
        h5py_ts = open("%s_ts_%s_%sSamp%s.hdf5" % (template_dir,str(idx),total_temp_num,tag),"rb")
        ts_new = h5py_ts['time_series/ts'][:]
        ts = np.vstack((ts,ts_new))

        # load corresponding parameters template pickle file
        h5py_par = open("%s_params_%s_%sSamp%s.hdf5" % (template_dir,str(idx),total_temp_num,tag),"rb")
        par_new = np.array(h5py_par['parameters/par'][:])
        par_new = np.reshape(par_new,(par_new.shape[0],1))
        par = np.vstack((par,par_new))

        print("loading file: _ts_%s_%sSamp.hdf5" % (str(idx),total_temp_num))
        print("loading file: _params_%s_%sSamp.hdf5" % (str(idx),total_temp_num))

        if idx < file_idx_list[-1]:
            ts = ts[:-1]
            par = par[:-1]

    ts = [ts]

    signal_train_images = np.reshape(ts[0], (ts[0].shape[0],ts[0].shape[2]))

    signal_train_pars = par

    # pick event-like signal as the true signal
    signal_image = signal_train_images[-1,:]
    signal_train_images = np.delete(signal_train_images,-1,axis=0)

    # add noise to signal
    noise_image = np.random.normal(0, n_sig, size=[1, signal_image.shape[0]])
    noise_signal = np.transpose(signal_image + noise_image)

    # define signal parameters
    signal_pars = signal_train_pars[-1,:]
    signal_train_pars = np.delete(signal_train_pars,-1,axis=0)

    #print(signal_train_images.shape, signal_train_pars.shape, signal_image.shape, noise_signal.shape, signal_pars.shape)
    #exit()


    return signal_train_images, signal_train_pars, signal_image, noise_signal, signal_pars

def overlap_tests(pred_samp,lalinf_samp,true_vals,kernel_cnn,kernel_lalinf):
    """ Perform Anderson-Darling, K-S, and overlap tests
    to get quantifiable values for accuracy of GAN
    PE method
    Parameters
    ----------
    pred_samp: numpy array
        predicted PE samples from CNN
    lalinf_samp: numpy array
        predicted PE samples from lalinference
    true_vals:
        true scalar point values for parameters to be estimated (taken from GW event paper)
    kernel_cnn: scipy kde instance
        gaussian kde of CNN results
    kernel_lalinf: scipy kde instance
        gaussian kde of lalinference results
    Returns
    -------
    ks_score:
        k-s test score
    ad_score:
        anderson-darling score
    beta_score:
        overlap score. used to determine goodness of CNN PE estimates
    """

    # do k-s test
    ks_mc_score = ks_2samp(pred_samp[:,0].reshape(pred_samp[:,0].shape[0],),lalinf_samp[0][:])
    ks_q_score = ks_2samp(pred_samp[:,1].reshape(pred_samp[:,1].shape[0],),lalinf_samp[1][:])
    ks_score = np.array([ks_mc_score,ks_q_score])

    # do anderson-darling test
    ad_mc_score = anderson_ksamp([pred_samp[:,0].reshape(pred_samp[:,0].shape[0],),lalinf_samp[0][:]])
    ad_q_score = anderson_ksamp([pred_samp[:,1].reshape(pred_samp[:,1].shape[0],),lalinf_samp[1][:]])
    ad_score = [ad_mc_score,ad_q_score]

    # compute overlap statistic
    comb_mc = np.concatenate((pred_samp[:,0].reshape(pred_samp[:,0].shape[0],1),lalinf_samp[0][:].reshape(lalinf_samp[0][:].shape[0],1)))
    comb_q = np.concatenate((pred_samp[:,1].reshape(pred_samp[:,1].shape[0],1),lalinf_samp[1][:].reshape(lalinf_samp[1][:].shape[0],1)))
    X, Y = np.mgrid[np.min(comb_mc):np.max(comb_mc):100j, np.min(comb_q):np.max(comb_q):100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    #cnn_pdf = np.reshape(kernel_cnn(positions).T, X.shape)
    cnn_pdf = kernel_cnn.pdf(positions)

    #X, Y = np.mgrid[np.min(lalinf_samp[0][:]):np.max(lalinf_samp[0][:]):100j, np.min(lalinf_samp[1][:]):np.max(lalinf_samp[1][:]):100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    #lalinf_pdf = np.reshape(kernel_lalinf(positions).T, X.shape)
    lalinf_pdf = kernel_lalinf.pdf(positions)

    beta_score = np.divide(np.sum( cnn_pdf*lalinf_pdf ),
                              np.sqrt(np.sum( cnn_pdf**2 ) * 
                              np.sum( lalinf_pdf**2 )))
    

    return ks_score, ad_score, beta_score

def main():
    # Set up data
    batch_size = 8 # set batch size
    test_split = 1000 # number of testing samples to use

    # load in gw templates and signals
    signal_train_images, signal_train_pars, signal_image, noise_signal, signal_pars = load_gw_data()

    for sig in signal_train_images:
        sig += np.random.normal(loc=0.0, scale=n_sig)

    # load in lalinference samples
    with open('gw_data/data/gw150914_mc_q_lalinf_post_srate-1024_python3.sav','rb' ) as f:
        lalinf_post = pickle.load(f) 
    lalinf_mc = lalinf_post[0]
    lalinf_q = lalinf_post[1]

    # declare gw variants of positions and labels
    labels = torch.tensor(signal_train_images, dtype=torch.float)
    pos = torch.tensor(signal_train_pars, dtype=torch.float)

    # setting up the model
    ndim_tot = 2048  # two times the number data dimensions?
    ndim_x = 2    # number of parameter dimensions
    ndim_y = 1024    # number of data dimensions
    ndim_z = 100    # number of latent space dimensions?

    # define different parts of the network
    # define input node
    inp = InputNode(ndim_tot, name='input')

    # define hidden layer nodes
    t1 = Node([inp.out0], rev_multiplicative_layer,
              {'F_class': F_fully_connected, 'clamp': 2.0,
               'F_args': {'dropout': 0.0}})

    t2 = Node([t1.out0], rev_multiplicative_layer,
              {'F_class': F_fully_connected, 'clamp': 2.0,
               'F_args': {'dropout': 0.0}})

    t3 = Node([t2.out0], rev_multiplicative_layer,
              {'F_class': F_fully_connected, 'clamp': 2.0,
               'F_args': {'dropout': 0.0}})

    # define output layer node
    outp = OutputNode([t3.out0], name='output')

    nodes = [inp, t1, t2, t3, outp]
    model = ReversibleGraphNet(nodes)

    # Train model
    # Training parameters
    n_epochs = 50000
    meta_epoch = 12 # what is this???
    n_its_per_epoch = 4
    batch_size = 32

    lr = 1e-2
    gamma = 0.01**(1./120)
    l2_reg = 2e-5

    y_noise_scale = 3e-2     # amount of noise to add to y parameter?
    zeros_noise_scale = 3e-2 # what is this??

    # relative weighting of losses:
    lambd_predict = 300. # forward pass
    lambd_latent = 300.  # laten space
    lambd_rev = 400.     # backwards pass

    # padding both the data and the latent space
    # such that they have equal dimension to the parameter space
    pad_x = torch.zeros(batch_size, ndim_tot - ndim_x)
    pad_yz = torch.zeros(batch_size, ndim_tot - ndim_y - ndim_z)


    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.8, 0.8),
                             eps=1e-04, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=meta_epoch,
                                                gamma=gamma)


    # define the three loss functions
    loss_backward = MMD_multiscale
    loss_latent = MMD_multiscale
    loss_fit = fit

    # set up test set data loader
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pos[:test_split], labels[:test_split]),
        batch_size=batch_size, shuffle=True, drop_last=True)

    # set up training set data loader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pos[test_split:], labels[test_split:]),
        batch_size=batch_size, shuffle=True, drop_last=True)


    # what is happening here? More set up of network?
    for mod_list in model.children():
        for block in mod_list.children():
            for coeff in block.children():
                coeff.fc3.weight.data = 0.01*torch.randn(coeff.fc3.weight.shape)
            
    model.to(device)

    # initialize gif for showing training procedure
    axes = plt.figure(1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Predicted labels (Forwards Process)')
    #axes[1].set_xticks([])
    #axes[1].set_yticks([])
    #axes[1].set_title('Generated Samples (Backwards Process)')
    #plt.show()
    #plt.canvas.draw()

    # number of test samples to use after training 
    N_samp = 4000

    # choose test samples to use after training
    # 1000 iterations of test signal burried in noise. Only need to change z parameter.
    #x_samps = torch.cat([x for x,y in test_loader], dim=0)[:N_samp]
    #y_samps = torch.cat([y for x,y in test_loader], dim=0)[:N_samp]
    #y_samps += torch.randn(N_samp, ndim_y) #* y_noise_scale
    y_samps = np.transpose(torch.tensor(np.repeat(noise_signal, N_samp, axis=1), dtype=torch.float))

    # make test samples. First element is the latent space dimension
    # second element is the extra zeros needed to pad the input.
    # the third element is the time series
    y_samps = torch.cat([torch.randn(N_samp, ndim_z),
                         torch.zeros(N_samp, ndim_tot - ndim_y - ndim_z), # zeros_noise_scale * 
                         y_samps], dim=1)
    # what we should have now are 1000 copies of the event burried in noise with zero padding up to 2048
    y_samps = y_samps.to(device)

    # start training loop            
    try:
    #     print('#Epoch \tIt/s \tl_total')
        t_start = time()
        # loop over number of epochs
        for i_epoch in tqdm(range(n_epochs), ascii=True, ncols=80):

            scheduler.step()

            # Initially, the l2 reg. on x and z can give huge gradients, set
            # the lr lower for this
            if i_epoch < 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * 1e-2

            #print(i_epoch, end='\t ')
            train(model,train_loader,n_its_per_epoch,zeros_noise_scale,batch_size,ndim_tot,ndim_x,ndim_y,ndim_z,y_noise_scale,optimizer,lambd_predict,loss_fit,lambd_latent,loss_latent,lambd_rev,loss_backward,i_epoch)

            rev_x = model(y_samps, rev=True)
            rev_x = rev_x.cpu().data.numpy()
        
            #pred_c = model(torch.cat((x_samps, torch.zeros(N_samp, ndim_tot - ndim_x)),
            #                         dim=1).to(device)).data[:, -8:].argmax(dim=1)

            #axes[0].clear()
            #axes[0].scatter(x_samps[:,0], x_samps[:,1], cmap='Set1', s=1., vmin=0, vmax=9)
            #axes[0].axis('equal')
            #axes[0].axis([-3,3,-3,3])
            #axes[0].set_xticks([])
            #axes[0].set_yticks([])

        
            #plt.clear()
            plt.scatter(rev_x[:,0], rev_x[:,1], s=1., c='red')
            plt.scatter(lalinf_mc, lalinf_q, s=1., c='blue')
            #axes[1].axis('equal')
            #axes[1].axis([-3,3,-3,3])
            #axes[1].set_xticks([])
            #axes[1].set_yticks([])
        
            #plt.canvas.draw()
            plt.savefig('/home/hunter.gabbard/public_html/ml_play_githubs/FrEIA/training_pred.png')
            plt.close()

    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nTraining took {(time()-t_start)/60:.2f} minutes\n")

main()
