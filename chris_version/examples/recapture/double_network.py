import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from loss import mse, mse_tv, mmd_multiscale_on
import torch
import chris_data as data_maker
import numpy as np

# class which defines forward model
class nn_double_f(nn.Sequential):
    """
    def __init__(self,size_in,size_out):
        super().__init__()
        self.in_lay = nn.Linear(size_in, 8)
        self.hidden1 = nn.Linear(8, 16)
        self.hidden2 = nn.Linear(16, 16)
        self.hidden3 = nn.Linear(16, 32)
        self.hidden4 = nn.Linear(32, 32)
        self.hidden5 = nn.Linear(32, 64)
        self.hidden6 = nn.Linear(64, 64)
        self.output = nn.Linear(64, size_out)
    """

    def __init__(self,size_in,size_out):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 6, 3)
        self.conv2 = nn.Conv1d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 15, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, size_out)
  
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        # If the size is a square you can only specify a single number
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        """
        x = self.in_lay(x)
        x = F.elu(x)
        x = self.hidden1(x)
        x = F.elu(x)
        x = self.hidden2(x)
        x = F.elu(x)
        x = self.hidden3(x)
        x = F.elu(x)
        x = self.hidden4(x)
        x = F.elu(x)
        x = self.hidden5(x)
        x = F.elu(x)
        x = self.hidden6(x)
        x = self.output(x)
        """
        return x

# class which defines reverse model
class nn_double_r(nn.Sequential):
    """
    def __init__(self,size_in,size_out):
        super().__init__()
        self.in_lay = nn.Linear(size_in, 8)
        self.hidden1 = nn.Linear(8, 16)
        self.hidden2 = nn.Linear(16, 16)
        self.hidden3 = nn.Linear(16, 32)
        self.hidden4 = nn.Linear(32, 32)
        self.hidden5 = nn.Linear(32, 64)
        self.hidden6 = nn.Linear(64, 64)
        self.output = nn.Linear(64, size_out)
    """
    def __init__(self,size_in,size_out):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, 3)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.conv3 = nn.Conv1d(16,32, 3)
        self.conv4 = nn.Conv1d(32,32, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(32 * 24, 64)  # 6*6 from image dimension
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, size_out)
    
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        # If the size is a square you can only specify a single number
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        """
        x = self.in_lay(x)
        x = F.elu(x)
        x = self.hidden1(x)
        x = F.elu(x)
        x = self.hidden2(x)
        x = F.elu(x)
        x = self.hidden3(x)
        x = F.elu(x)
        x = self.hidden4(x)
        x = F.elu(x)
        x = self.hidden5(x)
        x = F.elu(x)
        x = self.hidden6(x)
        x = self.output(x)
        """
        return x

# class which trains model
class DoubleNetTrainer:
    def __init__(self, model_f, model_r, atmosData, dev, load_model=False):
        self.model_f = model_f
        self.model_r = model_r
        self.atmosData = atmosData
        self.dev = dev
        self.mmFns = None

        self.model_f.to(dev)
        self.model_r.to(dev)

    def training_params(self, numEpochs, lr=2e-3, miniBatchesPerEpoch=20, metaEpoch=12, miniBatchSize=None,
                        l2Reg=2e-5, fadeIn=True,
                        loss_fit=None, loss_latent=None, loss_backward=None, ndata=128, sigma=0.2, seed=1,
                        batchSize=1600, usepars=[0,1,2]):
        if miniBatchSize is None:
            miniBatchSize = self.atmosData.batchSize

        if loss_latent is None:
            loss_latent = mmd_multiscale_on(self.dev)

        if loss_backward is None:
            loss_backward = mmd_multiscale_on(self.dev)

        decayEpochs = (numEpochs * miniBatchesPerEpoch) // metaEpoch
        gamma = 0.004**(1.0 / decayEpochs)

        # self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.8, 0.8),
        #                               eps=1e-06, weight_decay=l2Reg)
        self.optim_f = torch.optim.Adam(self.model_f.parameters(), lr=lr, betas=(0.8, 0.8),
                                      eps=1e-06, weight_decay=l2Reg)
        self.scheduler_f = torch.optim.lr_scheduler.StepLR(self.optim_f,
                                                         step_size=metaEpoch,
                                                         gamma=gamma)
        self.optim_r = torch.optim.Adam(self.model_r.parameters(), lr=lr, betas=(0.8, 0.8),
                                      eps=1e-06, weight_decay=l2Reg)
        self.scheduler_r = torch.optim.lr_scheduler.StepLR(self.optim_r,
                                                         step_size=metaEpoch,
                                                         gamma=gamma)
        self.fadeIn = fadeIn
        self.miniBatchSize = miniBatchSize
        self.miniBatchesPerEpoch = miniBatchesPerEpoch
        self.numEpochs = numEpochs
        self.loss_fit = loss_fit
        self.loss_latent = loss_latent
        self.loss_backward = loss_backward
        self.wRevScale = 1.0
        self.ndata = ndata
        self.sigma = sigma
        self.seed = seed
        self.batchSize = batchSize
        self.usepars = usepars

    def train(self, epoch, gen_inf_temp=False, extra_z=False, do_cnn=False):
        self.model_f.train()
        self.model_r.train()

        lTot = 0
        miniBatchIdx = 0

        randn = torch.randn(self.batchSize, self.ndata, dtype=torch.float, device=self.dev)

        optimizer_f = self.optim_f#optim.SGD(self.model_f.parameters(), lr=0.01, weight_decay= 1e-6, momentum = 0.9, nesterov = True)
        optimizer_r = self.optim_r#optim.SGD(self.model_r.parameters(), lr=0.01, weight_decay= 1e-6, momentum = 0.9, nesterov = True)

        losses = [0,0,0,0]

        # get data
        for x, y, y_sig in self.atmosData.trainLoader:
            miniBatchIdx += 1

            if miniBatchIdx > self.miniBatchesPerEpoch:
                break

            # if true, generate templates on the fly during training
            if gen_inf_temp:
                del x, y
                pos, labels, _, y_sig, _ = data_maker.generate(
                                                    tot_dataset_size=2*self.batchSize,
                                                    ndata=self.ndata,
                                                    usepars=self.usepars,
                                                    sigma=self.sigma,
                                                    seed=np.random.randint(int(1e9))
                                                    )
                loader = torch.utils.data.DataLoader(
                        torch.utils.data.TensorDataset(torch.tensor(pos), torch.tensor(labels), torch.tensor(y_sig)),
                        batch_size=self.batchSize, shuffle=True, drop_last=True)

                for x, y, y_sig in loader:
                    x = x
                    y = y
                    y_sig = y_sig
                    break

            n = y - y_sig
            x, y, y_sig, n = x.to(self.dev), y.to(self.dev), y_sig.to(self.dev), n.to(self.dev)
            yClean = y.clone()

            optimizer_f.zero_grad()
            optimizer_r.zero_grad()

            #################
            # forward process
            #################
            ## 1. forward propagation
            data = torch.cat((x[:],
                              n[:]), dim=1)
            if do_cnn: data = data.reshape(data.shape[0],1,data.shape[1])
            output = self.model_f(data)
            

            ## 2. loss calculation
            target = torch.cat((y[:],
                                randn), dim=1)

            loss_y = self.loss_fit(output[:,:y.shape[1]], y[:])

            ## 3. backward propagation
            loss_y.backward(retain_graph=True)
        
            losses[0]+=loss_y.data.item()

            loss_z = self.loss_latent(output[:,y.shape[1]:], randn)

            ## 3. backward propagation
            loss_z.backward()

            ## 4. weight optimization
            optimizer_f.step()

            losses[1]+=loss_z.data.item()

            #################
            # reverse process
            #################
            ## 1. forward propagation
            output= torch.cat((y[:],output[:,y.shape[1]:]),dim=1)
            if do_cnn: 
                output = output.reshape(output.shape[0],1,output.shape[1])
            output = self.model_r(output.data)

            ## 2. loss calculation
            target = torch.cat((x[:],
                                n[:]), dim=1)
            loss_r = self.loss_fit(output, target)

            ## 3. backward propagation
            loss_r.backward()

            ## 4. weight optimization
            optimizer_r.step()

            losses[2]+=loss_r.data.item()
            losses[3]+=0.0 # dummy loss for now
            lTot += losses[0] + losses[1] + losses[2] + losses[3]

        losses = [l / miniBatchIdx for l in losses]
        return lTot / miniBatchIdx, losses
            ## evaluation part 
            #model.eval()
            #for data, target in validloader:
            #    output = model(data)
            #    loss = loss_function(output, target)
            #    valid_loss.append(loss.item())
