import numpy as np
import torch
import torch.utils.data

def generate(tot_dataset_size,model='slope',ndata=8,sigma=0.1,prior_bound=[0,1,0,1],seed=0):

    np.random.seed(seed)
    N = tot_dataset_size

    if model=='slope':

        # draw gradient and intercept from prior
        pars = np.random.uniform(0,1,size=(N,2))
        pars[:,0] = prior_bound[0] + (prior_bound[1]-prior_bound[0])*pars[:,0]
        pars[:,1] = prior_bound[2] + (prior_bound[3]-prior_bound[2])*pars[:,1]

        # make y = mx + c + noise
        noise = np.random.normal(loc=0.0,scale=sigma,size=(N,ndata))
        xvec = np.arange(ndata)/float(ndata)
        sig = np.array([pars[:,0]*x + pars[:,1] for x in xvec]).transpose()
        data = sig + noise
        #data = np.array([pars[:,0]*x + pars[:,1] + n for x,n in zip(xvec,noise)]).transpose()

    elif model=='sg':

        # draw gradient and intercept from prior
        pars = np.random.uniform(0,1,size=(N,2))
        pars[:,0] = prior_bound[0] + (prior_bound[1]-prior_bound[0])*pars[:,0]
        pars[:,1] = prior_bound[2] + (prior_bound[3]-prior_bound[2])*pars[:,1]
        w = 6.0*np.pi
        p = 1.0
        tau = 0.25

        # make y = Asin()*exp()  + noise
        noise = np.random.normal(loc=0.0,scale=sigma,size=(N,ndata))
        xvec = np.arange(ndata)/float(ndata)
        sig = np.array([pars[:,0]*np.sin(w*x + p)*np.exp(-((x-pars[:,1])/tau)**2) for x in xvec]).transpose()
        data = sig + noise

    else:
        print('Sorry no model of that name')
        exit(1)

    # randomise the data 
    shuffling = np.random.permutation(N)
    pars = torch.tensor(pars[shuffling], dtype=torch.float)
    data = torch.tensor(data[shuffling], dtype=torch.float)
    sig = torch.tensor(sig[shuffling], dtype=torch.float)

    return pars, data, xvec, sig

def get_lik(ydata,n_grid=64,sig_model='sg',sigma=None,xvec=None,bound=[0,1,0,1]):

    mcx = np.linspace(bound[0],bound[1],n_grid)              # vector of mu values
    mcy = np.linspace(bound[2],bound[3],n_grid)
    dmcx = mcx[1]-mcx[0]                       # mu spacing
    dmcy = mcy[1]-mcy[0]
    mv, cv = np.meshgrid(mcx,mcy)        # combine into meshed variables

    res = np.zeros((n_grid,n_grid))
    if sig_model=='slope':
        for i,c in enumerate(mcy):
            res[i,:] = np.array([np.sum(((ydata-m*xvec-c)/sigma)**2) for m in mcx])
        res = np.exp(-0.5*res)
    elif sig_model=='sg':
        w = 6.0*np.pi
        p = 1.0
        tau = 0.25
        for i,t in enumerate(mcy):
            res[i,:] = np.array([np.sum(((ydata - A*np.sin(w*xvec + p)*np.exp(-((xvec-t)/tau)**2))/sigma)**2) for A in mcx])
        res = np.exp(-0.5*res)

    # normalise the posterior
    res /= (np.sum(res.flatten())*dmcx*dmcy)

    # compute integrated probability outwards from max point
    res = res.flatten()
    idx = np.argsort(res)[::-1]
    prob = np.zeros(n_grid*n_grid)
    prob[idx] = np.cumsum(res[idx])*dmcx*dmcy
    prob = prob.reshape(n_grid,n_grid)
    res = res.reshape(n_grid,n_grid)
    return mcx, mcy, prob


