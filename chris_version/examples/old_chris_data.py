import numpy as np
import torch
import torch.utils.data
import matplotlib.colors

verts = [
         (-2.4142, 1.),
         (-1., 2.4142),
         (1.,  2.4142),
         (2.4142,  1.),
         (2.4142, -1.),
         (1., -2.4142),
         (-1., -2.4142),
         (-2.4142, -1.)
        ]

label_maps = {
              'all':  [0, 1, 2, 3, 4, 5, 6, 7],
              'some': [0, 0, 0, 0, 1, 1, 2, 3],
              'none': [0, 0, 0, 0, 0, 0, 0, 0],
             }

rgb_maps = {
              0: [matplotlib.colors.to_rgb('b')],
              1: [matplotlib.colors.to_rgb('g')], 
              2: [matplotlib.colors.to_rgb('r')],
              3: [matplotlib.colors.to_rgb('c')],
              4: [matplotlib.colors.to_rgb('m')],
              5: [matplotlib.colors.to_rgb('y')],
              6: [matplotlib.colors.to_rgb('k')],
              7: [matplotlib.colors.to_rgb('w')],
           }

def generate(labels, tot_dataset_size):
    # print('Generating artifical data for setup "%s"' % (labels))

    np.random.seed(0)
    N = tot_dataset_size
    #mapping = label_maps[labels]

    #pos = np.random.normal(size=(N,3), scale=0.2)
    #labels = np.zeros((N, 8))
    labels = np.random.uniform(0,1,size=(N,2)) # mu and sig
    labels[:,1] = 0.5*(labels[:,1]+1.0)	# sig
    #n = N//8

    pos = np.random.normal(loc=0.0,scale=1.0,size=(N,3))
    pos = np.array([p*l + m for p,l,m in zip(pos,labels[:,1],labels[:,0])]).reshape(N,3)
    #pos = np.array([np.random.normal(loc=mu,scale=sig,size=3) for mu,sig in labels])

    #for i, v in enumerate(verts):
    #    pos[i*n:(i+1)*n, :] += v
    #    #labels[i*n:(i+1)*n, mapping[i]] = 1.
    #    c = np.array(rgb_maps[i]).flatten()
    #    labels[i*n:(i+1)*n,:] = c

    shuffling = np.random.permutation(N)
    pos = torch.tensor(pos[shuffling], dtype=torch.float)
    labels = torch.tensor(labels[shuffling], dtype=torch.float)

    return pos, labels

    # test_loader = torch.utils.data.DataLoader(
    #     torch.utils.data.TensorDataset(pos[:test_split], labels[:test_split]),
    #     batch_size=batch_size, shuffle=True, drop_last=True)

    # train_loader = torch.utils.data.DataLoader(
    #     torch.utils.data.TensorDataset(pos[test_split:], labels[test_split:]),
    #     batch_size=batch_size, shuffle=True, drop_last=True)

    # return test_loader, train_loader
