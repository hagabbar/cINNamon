import torch
import torch.optim

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import scipy
from scipy.stats import multivariate_normal as mvn
from scipy.special import logit, expit
from scipy.stats import uniform, gaussian_kde, ks_2samp, anderson_ksamp
from scipy import stats
from scipy.signal import butter, lfilter, freqs, resample

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import rev_multiplicative_layer, F_fully_connected, F_conv

import data as data

from sys import exit
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def train(model,train_loader,n_its_per_epoch,zeros_noise_scale,batch_size,ndim_tot,ndim_x,ndim_y,ndim_z,y_noise_scale,optimizer,lambd_predict,loss_fit,lambd_latent,loss_latent,lambd_rev,loss_backward,conv_nn,i_epoch=0):
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
        pad_x = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                ndim_x, device=device)
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)

        y += y_noise_scale * torch.randn(batch_size, ndim_y, dtype=torch.float, device=device)

        x, y = (torch.cat((x, pad_x),  dim=1),
                torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y),
                          dim=1))


        optimizer.zero_grad()

        # Forward step:
        if conv_nn == True: x = x.reshape(x.shape[0],x.shape[1],1,1)
        output = model(x)

        # Shorten output, and remove gradients wrt y, for latent loss
        y_short = torch.cat((y[:, :ndim_z], y[:, -ndim_y:]), dim=1)

        l = lambd_predict * loss_fit(output[:, ndim_z:], y[:, ndim_z:])
        l_forward = l

        output_block_grad = torch.cat((output[:, :ndim_z],
                                       output[:, -ndim_y:].data), dim=1)

        if conv_nn == True: output_block_grad=output_block_grad.reshape(output_block_grad.shape[0],output_block_grad.shape[1]) 
        l_latent = loss_latent(output_block_grad, y_short)
        l += lambd_latent * l_latent
        l_tot += l.data.item()

        l.backward()

        # Backward step:
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)
        y = y_clean + y_noise_scale * torch.randn(batch_size, ndim_y, device=device)

        if conv_nn == True: output = output.reshape(output.shape[0],output.shape[1])
        orig_z_perturbed = (output.data[:, :ndim_z] + y_noise_scale *
                            torch.randn(batch_size, ndim_z, device=device))
        y_rev = torch.cat((orig_z_perturbed, pad_yz,
                           y), dim=1)
        y_rev_rand = torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz,
                                y), dim=1)

        if conv_nn == True: y_rev=y_rev.reshape(y_rev.shape[0],y_rev.shape[1],1,1)
        if conv_nn == True: y_rev_rand=y_rev_rand.reshape(y_rev_rand.shape[0],y_rev_rand.shape[1],1,1)
        output_rev = model(y_rev, rev=True)
        output_rev_rand = model(y_rev_rand, rev=True)

        if conv_nn == True: output_rev=output_rev.reshape(output_rev.shape[0],output_rev.shape[1])
        if conv_nn == True: output_rev_rand=output_rev_rand.reshape(output_rev_rand.shape[0],output_rev_rand.shape[1])
        if conv_nn == True: x = x.reshape(x.shape[0],x.shape[1])
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

    return l_tot / batch_idx, l_latent, l_rev, l_forward

    target3 = norm*contour3

def make_contour_plot(ax,x,y,dataset,color='red',flip=False, kernel_lalinf=False, kernel_cnn=False):
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
    H,xedges,yedges=np.histogram2d(x,y,bins=20,normed=True)

    if flip == True:
        H,xedges,yedges=np.histogram2d(y,x,bins=20,normed=True)
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
    level4=H.max()
    levels=[level1,level2,level3,level4]

    # Pass levels to normed kde plot
    #sns.kdeplot(x,y,shade=True,ax=ax,n_levels=levels,cmap=color,alpha=0.5,normed=True)
    X, Y = np.mgrid[np.min(x):np.max(x):100j, np.min(y):np.max(y):100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    if not kernel_lalinf or not kernel_cnn: kernel = gaussian_kde(dataset)
    Z = np.reshape(kernel(positions).T, X.shape)
    ax.contour(X,Y,Z,levels=levels,alpha=0.5,colors=color)
    #ax.set_aspect('equal')

    return kernel

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
    #print(positions.shape,pred_samp.shape)
    cnn_pdf = kernel_cnn.pdf(positions)

    #X, Y = np.mgrid[np.min(lalinf_samp[0][:]):np.max(lalinf_samp[0][:]):100j, np.min(lalinf_samp[1][:]):np.max(lalinf_samp[1][:]):100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    #lalinf_pdf = np.reshape(kernel_lalinf(positions).T, X.shape)
    lalinf_pdf = kernel_lalinf.pdf(positions)

    beta_score = np.divide(np.sum( cnn_pdf*lalinf_pdf ),
                              np.sqrt(np.sum( cnn_pdf**2 ) * 
                              np.sum( lalinf_pdf**2 )))
    

    return ks_score, ad_score, beta_score

def plot_losses(losses,filename,logscale=False,legend=None):
    """ Make loss and accuracy plots and output to file.
    Plot with x and y log-axes is desired
    Parameters
    ----------
    losses: list
        list containing history of network loss and accuracy values
    filename: string
        string which specifies location of output directory and filename
    logscale: boolean
        if True: use logscale in plots, if False: do not use
    legend: boolean
        if True: apply legend, if False: do not
    pe_losses = [losstot_hist, losslatent_hist, lossrev_hist, lossf_hist]
    """
    # plot forward pass loss
    fig = plt.figure()
    losses = np.array(losses)
    ax1 = fig.add_subplot(411)	
    ax1.plot(losses[0],'b', label='Total')
    ax1.set_xlabel(r'epoch')
    ax1.set_ylabel(r'loss')
    if legend is not None:
    	ax1.legend(loc='upper left')
    
    # plot backward pass loss
    ax2 = fig.add_subplot(412)
    ax2.plot(losses[1],'r', label='latent')
    # rescale axis using a logistic function so that we see more detail
    # close to 0 and close 1
    ax2.set_xlabel(r'epoch')
    ax2.set_ylabel(r'loss')
    if legend is not None:
        ax2.legend(loc='upper left')

    ax3 = fig.add_subplot(413)
    ax3.plot(losses[2],'g', label='reversible')
    # rescale axis using a logistic function so that we see more detail
    # close to 0 and close 1
    ax3.set_xlabel(r'epoch')
    ax3.set_ylabel(r'loss')
    if legend is not None:
        ax3.legend(loc='upper left')

    ax4 = fig.add_subplot(414)
    ax4.plot(losses[3],'cyan', label='forward')
    # rescale axis using a logistic function so that we see more detail
    # close to 0 and close 1
    ax4.set_xlabel(r'epoch')
    ax4.set_ylabel(r'loss')
    if legend is not None:
        ax4.legend(loc='upper left')

    if logscale==True:
        ax1.set_xscale("log", nonposx='clip')
        ax1.set_yscale("log", nonposy='clip')
        ax2.set_xscale("log", nonposx='clip')
        ax2.set_yscale("log", nonposy='clip')
        ax3.set_xscale("log", nonposx='clip')
        ax3.set_yscale("log", nonposy='clip')
        ax4.set_xscale("log", nonposx='clip')
        ax4.set_yscale("log", nonposy='clip')
    plt.savefig(filename)
    plt.close('all')


def main():

    # Set up simulation parameters
    batch_size = 1600  # set batch size
    r = 3              # the grid dimension for the output tests
    test_split = r*r   # number of testing samples to use
    sig_model = 'sg'   # the signal model to use
    sigma = 0.2        # the noise std
    ndata = 8         #32 number of data samples in time series
    bound = [0.0,1.0,0.0,1.0]         # effective bound for likelihood
    seed = 1           # seed for generating data
    out_dir = "/home/hunter.gabbard/public_html/CBC/cINNamon/gausian_results/"
    n_neurons = 0
    do_contours = True # if True, plot contours of predictions by INN
    plot_cadence = 25
    do_latent_struc = False # if True, plot latent space 2D structure
    conv_nn = True # if True, use convolutional nn structure

    # setup output directory - if it does not exist
    os.system('mkdir -p %s' % out_dir)

    # generate data
    pos, labels, x, sig = data.generate(
        model=sig_model,
        tot_dataset_size=int(1e6), # 1e6
        ndata=ndata,
        sigma=sigma,
        prior_bound=bound,
        seed=seed
    )

    if do_latent_struc:
        # calculate mode of x-space for both pars
        mode_1 = stats.mode(np.array(pos[:,0]))
        mode_2 = stats.mode(np.array(pos[:,1]))

    # seperate the test data for plotting
    pos_test = pos[-test_split:]
    labels_test = labels[-test_split:]
    sig_test = sig[-test_split:]

    # plot the test data examples
    plt.figure(figsize=(6,6))
    fig_post, axes = plt.subplots(r,r,figsize=(6,6))
    cnt = 0
    for i in range(r):
        for j in range(r):
            axes[i,j].plot(x,np.array(labels_test[cnt,:]),'.')
            axes[i,j].plot(x,np.array(sig_test[cnt,:]),'-')
            cnt += 1
            axes[i,j].axis([0,1,-1.5,1.5])
    plt.savefig("%stest_distribution.png" % out_dir,dpi=360)
    plt.close()

    # setting up the model 
    ndim_x = 2        # number of posterior parameter dimensions (x,y)
    ndim_y = ndata    # number of label dimensions (noisy data samples)
    ndim_z = 2        # number of latent space dimensions?
    ndim_tot = max(ndim_x,ndim_y+ndim_z) + n_neurons     # must be > ndim_x and > ndim_y + ndim_z

    # define different parts of the network
    # define input node
    inp = InputNode(ndim_tot, name='input')

    # define hidden layer nodes
    if conv_nn == True:
        t1 = Node([inp.out0], rev_multiplicative_layer,
                  {'F_class': F_conv, 'clamp': 2.0,
                   'F_args':{}})

        t2 = Node([t1.out0], rev_multiplicative_layer,
                  {'F_class': F_conv, 'clamp': 2.0,
                   'F_args':{}})

        t3 = Node([t2.out0], rev_multiplicative_layer,
                  {'F_class': F_conv, 'clamp': 2.0,
                   'F_args':{}})

    # define output layer node
    outp = OutputNode([t3.out0], name='output')

    nodes = [inp, t1, t2, t3, outp]
    model = ReversibleGraphNet(nodes)

    # Train model
    # Training parameters
    n_epochs = 10000
    meta_epoch = 12 # what is this???
    n_its_per_epoch = 12
    batch_size = 1600

    lr = 1e-2
    gamma = 0.01**(1./120)
    l2_reg = 2e-5

    y_noise_scale = 3e-2
    zeros_noise_scale = 3e-2

    # relative weighting of losses:
    lambd_predict = 300. # forward pass
    lambd_latent = 300.  # laten space
    lambd_rev = 400.     # backwards pass

    # padding both the data and the latent space
    # such that they have equal dimension to the parameter space
    #pad_x = torch.zeros(batch_size, ndim_tot - ndim_x)
    #pad_yz = torch.zeros(batch_size, ndim_tot - ndim_y - ndim_z)

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

    # set up training set data loader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pos[test_split:], labels[test_split:]),
        batch_size=batch_size, shuffle=True, drop_last=True)

    # initialisation of network weights
    for mod_list in model.children():
        for block in mod_list.children():
            for coeff in block.children():
                if conv_nn == True: coeff.conv3.weight.data = 0.01*torch.randn(coeff.conv3.weight.shape)
    model.to(device)

    # number of test samples to use after training 
    N_samp = 2500

    # precompute true likelihood on the test data
    Ngrid = 64 
    cnt = 0
    lik = np.zeros((r,r,Ngrid*Ngrid))
    true_post = np.zeros((r,r,N_samp,2))
    lossf_hist = []
    lossrev_hist = []
    losstot_hist = []
    losslatent_hist = []
    beta_score_hist = []

    for i in range(r):
        for j in range(r):
            mvec,cvec,temp,post_points = data.get_lik(np.array(labels_test[cnt,:]).flatten(),n_grid=Ngrid,sig_model=sig_model,sigma=sigma,xvec=x,bound=bound)
            lik[i,j,:] = temp.flatten()
            true_post[i,j,:] = post_points[:N_samp]
            cnt += 1

    # start training loop
    try:
        t_start = time()
        # loop over number of epochs
        for i_epoch in tqdm(range(n_epochs), ascii=True, ncols=80):

            scheduler.step()

            # Initially, the l2 reg. on x and z can give huge gradients, set
            # the lr lower for this
            if i_epoch < 0:
                print('inside this iepoch<0 thing')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * 1e-2

            # train the model
            losstot, losslatent, lossrev, lossf = train(model,train_loader,n_its_per_epoch,zeros_noise_scale,batch_size,
                ndim_tot,ndim_x,ndim_y,ndim_z,y_noise_scale,optimizer,lambd_predict,
                loss_fit,lambd_latent,loss_latent,lambd_rev,loss_backward,conv_nn,i_epoch)

            # append current loss value to loss histories
            lossf_hist.append(lossf.data.item())
            lossrev_hist.append(lossrev.data.item())
            losstot_hist.append(losstot)
            losslatent_hist.append(losslatent.data.item())
            pe_losses = [losstot_hist, losslatent_hist, lossrev_hist, lossf_hist]

            # loop over a few cases and plot results in a grid
            cnt = 0
            beta_max = 0
            if ((i_epoch % plot_cadence == 0) & (i_epoch>0)):
                # use the network to predict parameters\
                
                if do_latent_struc:
                    # do latent space structure plotting
                    y_samps_latent = np.tile(np.array(labels_test[0,:]),1).reshape(1,ndim_y)
                    y_samps_latent = torch.tensor(y_samps_latent, dtype=torch.float)
                    x1_i_dist = []
                    x2_i_dist = []
                    x1_i_par = np.array([])
                    x2_i_par = np.array([])

                    # define latent space mesh grid
                    z_mesh = np.mgrid[-0.99:-0.01:100j, -0.99:-0.01:100j]
                    z_mesh = np.vstack([z_mesh,np.zeros((2,100,100))])

                    #for z_i in range(10000):
                    for i in range(z_mesh.shape[1]):
                        for j in range(z_mesh.shape[2]):
                            a = torch.randn(1,ndim_z)
                            a[0,0] = z_mesh[0,i,j]
                            a[0,1] = z_mesh[1,i,j]
                            x_i = model(torch.cat([a, 
                            torch.zeros(1, ndim_tot - ndim_y - ndim_z),
                            y_samps_latent], dim=1).to(device), rev=True)
                            x_i = x_i.cpu().data.numpy()
                    
                            # calculate hue and intensity
                            if np.abs(mode_1[0][0] - x_i[0][0]) < np.abs(mode_2[0][0] - x_i[0][1]):
                                z_mesh[2,i,j] = np.abs(mode_1[0][0] - x_i[0][0])
                                z_mesh[3,i,j] = 0

                            else:
                                z_mesh[2,i,j] = np.abs(mode_2[0][0] - x_i[0][1])
                                z_mesh[3,i,j] = 1 
                        
                    z_mesh[2,:,:][z_mesh[3,:,:] == 0] = z_mesh[2,:,:][z_mesh[3,:,:] == 0] / np.max(z_mesh[2,:,:][z_mesh[3,:,:] == 0])
                    z_mesh[2,:,:][z_mesh[3,:,:] == 1] = z_mesh[2,:,:][z_mesh[3,:,:] == 1] / np.max(z_mesh[2,:,:][z_mesh[3,:,:] == 1])

                    bg_color = 'black'
                    fg_color = 'red'

                    fig = plt.figure(facecolor=bg_color, edgecolor=fg_color)
                    axes = fig.add_subplot(111)
                    axes.patch.set_facecolor(bg_color)
                    axes.xaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
                    axes.yaxis.set_tick_params(color=fg_color, labelcolor=fg_color)
                    for spine in axes.spines.values():
                        spine.set_color(fg_color)
                    plt.scatter(z_mesh[0,:,:][z_mesh[3,:,:] == 0], z_mesh[1,:,:][z_mesh[3,:,:] == 0], s=1, c=z_mesh[2,:,:][z_mesh[3,:,:] == 0], cmap='Greens', axes=axes)
                    plt.scatter(z_mesh[0,:,:][z_mesh[3,:,:] == 1], z_mesh[1,:,:][z_mesh[3,:,:] == 1], s=1, c=z_mesh[2,:,:][z_mesh[3,:,:] == 1], cmap='Purples', axes=axes)
                    plt.xlabel('z-space', color=fg_color)
                    plt.ylabel('z-space', color=fg_color)
                    plt.savefig('%sstruct_z.png' % out_dir, dpi=360)
                    plt.close()

                # end of latent space structure plotting
                

                # initialize plot for showing testing results
                fig, axes = plt.subplots(r,r,figsize=(6,6))
                for i in range(r):
                    for j in range(r):

                        # convert data into correct format
                        y_samps = np.tile(np.array(labels_test[cnt,:]),N_samp).reshape(N_samp,ndim_y)
                        y_samps = torch.tensor(y_samps, dtype=torch.float)
                        #y_samps += y_noise_scale * torch.randn(N_samp, ndim_y)
                        y_samps = torch.cat([torch.randn(N_samp, ndim_z), #zeros_noise_scale * 
                            torch.zeros(N_samp, ndim_tot - ndim_y - ndim_z),
                            y_samps], dim=1)
                        y_samps = y_samps.to(device)

                        if conv_nn == True: y_samps = y_samps.reshape(y_samps.shape[0],y_samps.shape[1],1,1)
                        rev_x = model(y_samps, rev=True)
                        rev_x = rev_x.cpu().data.numpy()

                        if conv_nn == True: rev_x = rev_x.reshape(rev_x.shape[0],rev_x.shape[1])
 
                        # plot the samples and the true contours
                        axes[i,j].clear()
                        axes[i,j].contour(mvec,cvec,lik[i,j,:].reshape(Ngrid,Ngrid),levels=[0.68,0.9,0.99])
                        axes[i,j].scatter(rev_x[:,0], rev_x[:,1],s=0.5,alpha=0.5,color='red')
                        axes[i,j].scatter(true_post[i,j,:,1],true_post[i,j,:,0],s=0.5,alpha=0.5,color='blue')
                        axes[i,j].plot(pos_test[cnt,0],pos_test[cnt,1],'+r',markersize=8)
                        axes[i,j].axis(bound)

                        # add contours to results
                        if do_contours:
                            contour_y = np.reshape(rev_x[:,1], (rev_x[:,1].shape[0]))
                            contour_x = np.reshape(rev_x[:,0], (rev_x[:,0].shape[0]))
                            contour_dataset = np.array([contour_x,contour_y])
                            kernel_cnn = make_contour_plot(axes[i,j],contour_x,contour_y,contour_dataset,'red',flip=False, kernel_cnn=False)
                     
                            # run overlap tests on results
                            contour_x = np.reshape(true_post[i,j][:,1], (true_post[i,j][:,1].shape[0]))
                            contour_y = np.reshape(true_post[i,j][:,0], (true_post[i,j][:,0].shape[0]))
                            contour_dataset = np.array([contour_x,contour_y])
                            ks_score, ad_score, beta_score = overlap_tests(rev_x,true_post[i,j],pos_test[cnt],kernel_cnn,gaussian_kde(contour_dataset))
                            axes[i,j].legend(['Overlap: %s' % str(np.round(beta_score,3))])    
                            
                            beta_score_hist.append([beta_score])

                        cnt += 1

                # sve the results to file
                fig_post.canvas.draw()
                plt.savefig('%sposteriors_%s.png' % (out_dir,i_epoch),dpi=360)
                plt.savefig('%slatest.png' % out_dir,dpi=360)

                plot_losses(pe_losses,'%spe_losses.png' % out_dir,legend=['PE-GEN'])
                plot_losses(pe_losses,'%spe_losses_logscale.png' % out_dir,logscale=True,legend=['PE-GEN'])

    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nTraining took {(time()-t_start)/60:.2f} minutes\n")

main()
