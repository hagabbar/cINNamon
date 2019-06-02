import torch
from torch import nn
import numpy as np

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import rev_multiplicative_layer, permute_layer

from loss import mse, mse_tv, mmd_multiscale_on

from scipy.interpolate import interp1d
import h5py

from copy import deepcopy
from itertools import accumulate
import pickle
import chris_data as data_maker

from sys import exit

PadOp = '!!PAD'
ZeroPadOp = '!!ZeroPadding'

def schema_min_len(schema, zeroPadding):
    length = sum(s[1] if s[0] != PadOp else 0 for s in schema) \
            + zeroPadding * (len([s for s in schema if s[0] != PadOp]) - 1)
    return length
        
class DataSchema1D:
    def __init__(self, inp, minLength, zeroPadding, zero_pad_fn=torch.zeros):
        self.zero_pad = zero_pad_fn
        # Check schema is valid
        padCount = sum(1 if i[0] == PadOp else 0 for i in inp)
        for i in range(len(inp)-1):
            if inp[i][0] == PadOp and inp[i+1][0] == PadOp:
                raise ValueError('Schema cannot contain two consecutive \'!!PAD\' instructions.')
        # if padCount > 1:
        #     raise ValueError('Schema can only contain one \'!!PAD\' instruction.')
        if len([i for i in inp if i[0] != PadOp]) > len(set([i[0] for i in inp if i[0] != PadOp])):
            raise ValueError('Schema names must be unique within a schema.')
        
        # Find length without extra padding (beyond normal channel separation)
        length = schema_min_len(inp, zeroPadding)
        if (minLength - length) // padCount != (minLength - length) / padCount:
            raise ValueError('Schema padding isn\'t divisible by number of PadOps')

        # Build schema
        schema = []
        padding = (ZeroPadOp, zeroPadding)
        for j, i in enumerate(inp):
            if i[0] == PadOp:
                if j == len(inp) - 1:
                    # Count the edge case where '!!PAD' is the last op and a spurious
                    # extra padding gets inserted before it
                    if schema[-1] == padding:
                        del schema[-1]

                if length < minLength:
                    schema.append((ZeroPadOp, (minLength - length) // padCount))
                continue

            schema.append(i)
            if j != len(inp) - 1:
                schema.append(padding)

        if padCount == 0 and length < minLength:
            schema.append((ZeroPadOp, minLength - length))
        
        # Fuse adjacent zero padding -- no rational way to have more than two in a row 
        fusedSchema = []
        i = 0
        while True:
            if i >= len(schema):
                break

            if i < len(schema) - 1  and schema[i][0] == ZeroPadOp and schema[i+1][0] == ZeroPadOp:
                fusedSchema.append((ZeroPadOp, schema[i][1] + schema[i+1][1]))
                i += 1
            else:
                fusedSchema.append(schema[i])
            i += 1
        # Also remove 0-width ZeroPadding
        fusedSchema = [s for s in fusedSchema if s != (ZeroPadOp, 0)]
        self.schema = fusedSchema
        schemaTags = [s[0] for s in self.schema if s[0] != ZeroPadOp]
        tagIndices = [0] + list(accumulate([s[1] for s in self.schema]))
        tagRange = [(s[0], range(tagIndices[i], tagIndices[i+1])) for i, s in enumerate(self.schema) if s[0] != ZeroPadOp]
        for name, r in tagRange:
            setattr(self, name, r)
        self.len = tagIndices[-1]

    def __len__(self):
        return self.len

    def fill(self, entries, zero_pad_fn=None, batchSize=None, checkBounds=False, dev='cpu'):
        # Try and infer batchSize
        if batchSize is None:
            for k, v in entries.items():
                if not callable(v):
                    batchSize = v.shape[0]
                    break
            else:
                raise ValueError('Unable to infer batchSize from entries (all fns?). Set batchSize manually.')
        
        if checkBounds:
            try:
                for s in self.schema:
                    if s[0] == ZeroPadOp:
                        continue
                    entry = entries[s[0]]
                    if not callable(entry):
                        if len(entry.shape) != 2:
                            raise ValueError('Entry: %s must be a 2D array or fn.' % s[0])
                        if entry.shape[0] != batchSize:
                            raise ValueError('Entry: %s does not match batchSize along dim=0.' % s[0]) 
                        if entry.shape[1] != s[1]:
                            raise ValueError('Entry: %s does not match schema dimension.' % s[0]) 
            except KeyError as e:
                raise ValueError('No key present in entries to schema: ' + repr(e))
         
        # Use different zero_pad if specified
        if zero_pad_fn is None:
             zero_pad_fn = self.zero_pad
        
        # Fill in the schema, throw exception if entry is missing
        reifiedSchema = []
        try:
            for s in self.schema:
                if s[0] == ZeroPadOp:
                    reifiedSchema.append(zero_pad_fn(batchSize, s[1]))
                else:
                    entry = entries[s[0]]
                    if callable(entry):
                        reifiedSchema.append(entry(batchSize, s[1]))
                    else:
                        if s[0] == 'amp' or s[0] == 't0' or s[0] == 'tau': 
                            entry = entry.reshape(entry.shape[0],1)
                        reifiedSchema.append(entry)
        except KeyError as e:
            raise ValueError('No key present in entries to schema: ' + repr(e))

        reifiedSchema = torch.cat(reifiedSchema, dim=1)
        return reifiedSchema

    def __repr__(self):
        return repr(self.schema)

class F_fully_connected_leaky(nn.Module):
    '''Fully connected tranformation, not reversible, but used below.'''

    def __init__(self, size_in, size, internal_size=None, dropout=0.0,
                 batch_norm=False, leaky_slope=0.01):
        super(F_fully_connected_leaky, self).__init__()
        if not internal_size:
            internal_size = 2*size

        self.d1 = nn.Dropout(p=dropout)
        self.d2 = nn.Dropout(p=dropout)
        self.d2b = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(size_in, internal_size)
        self.fc2 = nn.Linear(internal_size, internal_size)
        self.fc2b = nn.Linear(internal_size, internal_size)
        self.fc2c = nn.Linear(internal_size, internal_size)
        self.fc2d  = nn.Linear(internal_size, internal_size)
        self.fc3 = nn.Linear(internal_size, size)

        self.nl1 = nn.ReLU()
        self.nl2 = nn.LeakyReLU(negative_slope=leaky_slope)
        self.nl2b = nn.LeakyReLU(negative_slope=leaky_slope)
        # self.nl1 = nn.ReLU()
        # self.nl2 = nn.ReLU()
        # self.nl2b = nn.ReLU()
        self.nl2c = nn.LeakyReLU(negative_slope=leaky_slope)
        self.nl2d = nn.ReLU()

        if batch_norm:
            self.bn1 = nn.BatchNorm1d(internal_size)
            self.bn1.weight.data.fill_(1)
            self.bn2 = nn.BatchNorm1d(internal_size)
            self.bn2.weight.data.fill_(1)
            self.bn2b = nn.BatchNorm1d(internal_size)
            self.bn2b.weight.data.fill_(1)
        self.batch_norm = batch_norm

    def forward(self, x):
        out = self.fc1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.nl1(self.d1(out))

        out = self.fc2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.nl2(self.d2(out))

        out = self.fc2b(out)
        if self.batch_norm:
            out = self.bn2b(out)
        out = self.nl2b(self.d2b(out))

        out = self.fc2c(out)
        out = self.nl2c(out)

        out = self.fc2d(out)
        out = self.nl2d(out)

        out = self.fc3(out)
        return out

class RadynversionNet(ReversibleGraphNet):
    def __init__(self, inputs, outputs, zeroPadding=0, numInvLayers=5, dropout=0.00, minSize=None):
        # Determine dimensions and construct DataSchema
        inMinLength = schema_min_len(inputs, zeroPadding)
        outMinLength = schema_min_len(outputs, zeroPadding)
        minLength = max(inMinLength, outMinLength)
        if minSize is not None:
            minLength = max(minLength, minSize)
        self.inSchema = DataSchema1D(inputs, minLength, zeroPadding)
        self.outSchema = DataSchema1D(outputs, minLength, zeroPadding)
        if len(self.inSchema) != len(self.outSchema):
            raise ValueError('Input and output schemas do not have the same dimension.')

        # Build net graph
        inp = InputNode(len(self.inSchema), name='Input (0-pad extra channels)')
        nodes = [inp]

        for i in range(numInvLayers):
            nodes.append(Node([nodes[-1].out0], rev_multiplicative_layer,
                         {'F_class': F_fully_connected_leaky, 'clamp': 2.0,
                          'F_args': {'dropout': 0.0}}, name='Inv%d' % i))
            if (i != numInvLayers - 1):
                nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': i}, name='Permute%d' % i))

        nodes.append(OutputNode([nodes[-1].out0], name='Output'))
        # Build net
        super().__init__(nodes)


class RadynversionTrainer:
    def __init__(self, model, atmosData, dev, load_model=False):
        self.model = model
        self.atmosData = atmosData
        self.dev = dev
        self.mmFns = None

        if not load_model:
            for mod_list in model.children():
                for block in mod_list.children():
                    for coeff in block.children():
                        coeff.fc3.weight.data = 1e-3*torch.randn(coeff.fc3.weight.shape)
#                        coeff.fc3.weight.data = 1e-2*torch.randn(coeff.fc3.weight.shape)

        self.model.to(dev)

    def cov(self,m, y=None):
        """
        computes the covariance matrix for a 2 channel tensor
        matches the behaviour of np.cov
        input DxN output DxD
        """
        if y is not None:
            m = torch.cat((m, y), dim=0)
        m_exp = torch.mean(m, dim=1)
        # removed mean because it was causing network to predict waveforms with same noise realization
        x = m #- m_exp[:, None]
        cov = 1 / (x.size(1) - 1) * x.mm(x.t())
        return cov

    def training_params(self, numEpochs, lr=2e-3, miniBatchesPerEpoch=20, metaEpoch=12, miniBatchSize=None, 
                        l2Reg=2e-5, wPred=1500, wLatent=300, wRev=500, zerosNoiseScale=5e-3, fadeIn=True,
                        loss_fit=mse, loss_latent=None, loss_backward=None, ndata=128, sigma=0.2, seed=1, 
                        n_neurons=0, batchSize=1600, usepars=[0,1,2], y_noise_scale=5e-2):
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
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.8, 0.8),
                                      eps=1e-06, weight_decay=l2Reg)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim,
                                                         step_size=metaEpoch,
                                                         gamma=gamma)
        self.wPred = wPred
        self.fadeIn = fadeIn
        self.wLatent = wLatent
        self.wRev = wRev
        self.zerosNoiseScale = zerosNoiseScale
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
        self.n_neurons = n_neurons
        self.y_noise_scale = y_noise_scale

    def train(self, epoch, gen_inf_temp=False, extra_z=False, do_covar=False):
        self.model.train()

        lTot = 0
        miniBatchIdx = 0
        if self.fadeIn:
            # normally at 0.4
            wRevScale = min(epoch / 400.0, 1)**3
            self.wRevScale = wRevScale
        else:
            wRevScale = 1.0
            self.wRevScale = wRevScale
        noiseScale = (1.0 - wRevScale) * self.zerosNoiseScale

        pad_fn = lambda *x: noiseScale * torch.randn(*x, device=self.dev) #+ 10 * torch.ones(*x, device=self.dev)
        randn = lambda *x: torch.randn(*x, device=self.dev)
        losses = [0, 0, 0, 0]

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

            # loss factors
            loss_factor_fwd_mmd_z = 1.0 #min(epoch / 400.0, 1)**3
            loss_factor_rev_mse_n = min(epoch / 500.0, 1)**3 # 1000
            loss_factor_fwd_mse_y = min(epoch / 750.0, 1)**3  # 2000
            loss_factor_rev_mse_x = min(epoch / 10000.0, 1)**3 # 3000

            if extra_z:
                xzp = self.model.inSchema.fill({'amp': x[:, 0],
                                           't0': x[:, 1],
                                           'tau': x[:, 2],
                                           'yNoise': n[:]},
                                          zero_pad_fn=pad_fn)
            else: 
                xp = self.model.inSchema.fill({'amp': x[:, 0], 
                                           't0': x[:, 1], 
                                           'tau': x[:, 2]},
                                          zero_pad_fn=pad_fn)
            if self.y_noise_scale:
                y += self.y_noise_scale * torch.randn(self.batchSize, self.ndata, dtype=torch.float, device=self.dev)

            yzp = self.model.outSchema.fill({'timeseries': y[:], 
                                             'LatentSpace': randn},
                                            zero_pad_fn=pad_fn)
            y_sig_zp = self.model.outSchema.fill({'timeseries': y_sig[:],
                                             'LatentSpace': randn},
                                            zero_pad_fn=pad_fn)
            yzpRevRand = self.model.outSchema.fill({'timeseries': yClean[:],
                                                    'LatentSpace': randn},
                                                   zero_pad_fn=pad_fn)

            self.optim.zero_grad()

            if extra_z: out= self.model(xzp)
            else: out = self.model(xp)

            # lForward = self.wPred * (self.loss_fit(y[:, 0], out[:, self.model.outSchema.Halpha]) + 
            #                          self.loss_fit(y[:, 1], out[:, self.model.outSchema.Ca8542]))
#             lForward = self.wPred * self.loss_fit(yzp[:, :self.model.outSchema.LatentSpace[0]], out[:, :self.model.outSchema.LatentSpace[0]])

            # add z space onto x-space
            #if extra_z:
            #    out_fmse = torch.cat((self.model(yzpRevRand, rev=True)[:, self.model.inSchema.LatentSpace], xzp[:, self.model.outSchema.LatentSpace[-1]+1:]),
            #                          dim=1)
            #    out_fmse = self.model(out_fmse)
            #    lForward = self.wPred * self.loss_fit(yzp[:, self.model.outSchema.LatentSpace[-1]+1:],
            #                                          out_fmse[:, self.model.outSchema.LatentSpace[-1]+1:])
            # use a covariance loss on forward mean squared error
            if do_covar:
                # try covariance fit
                output_cov = self.cov((out[:, self.model.outSchema.LatentSpace[-1]+1:]-y_sig_zp[:, self.model.outSchema.LatentSpace[-1]+1:]).transpose(0,1))
                ycov_mat = self.sigma*self.sigma*torch.eye((self.ndata+self.n_neurons),device=self.dev)

                lForward = self.wPred * self.loss_fit(output_cov.flatten(),
                                                      ycov_mat.flatten())

            else:
                # compute mean squared error on only y
                lForward = loss_factor_fwd_mse_y * self.wPred * self.loss_fit(yzp[:, self.model.outSchema.LatentSpace[-1]+1:], 
                                                      out[:, self.model.outSchema.LatentSpace[-1]+1:])
            losses[0] += lForward.data.item() / self.wPred
            #lForward_extra = self.wPred * self.loss_fit(yzp[:, self.model.outSchema.LatentSpace[-1]+1:],
            #                                          out[:, self.model.outSchema.LatentSpace[-1]+1:])
            #lForward += lForward_extra 
            #losses[0] += lForward_extra.data.item() / self.wPred            

            #if do_covar:
            do_z_covar=False
            if do_z_covar:
                # compute MMD loss on forward time series prediction
                lforward21Pred=out[:, self.model.outSchema.timeseries].data
                lforward21Target=yzp[:, self.model.outSchema.timeseries]
                lForward21 = self.wLatent * self.loss_latent(lforward21Pred, lforward21Target)

                losses[1] += lForward21.data.item() / self.wLatent
                lForward += lForward21

                # compute variance loss on z (2nd moment)
                lforward22Pred = self.cov((out[:, self.model.outSchema.LatentSpace]).transpose(0,1))
                lforward22Target = 1.0 * torch.eye((out[:, self.model.outSchema.LatentSpace].shape[1]),device=self.dev)

                lForward22 = self.wLatent * self.loss_fit(lforward22Pred.flatten(),
                                                        lforward22Target.flatten())
                
                losses[1] += lForward22.data.item() / self.wLatent
                lForward += lForward22

                # compute mean loss (1st moment) on z
                lForwardFirstMom = self.wLatent * self.loss_fit(torch.mean(out[:, self.model.outSchema.LatentSpace]),
                                                                torch.tensor(0.0))

                losses[1] += lForwardFirstMom.data.item() / self.wLatent
                lForward += lForwardFirstMom

                # compute 3rd moment loss on z
                lForwardThirdMom = self.wLatent * self.loss_fit(torch.mean(out[:, self.model.outSchema.LatentSpace])**3 + 3*torch.mean(out[:, self.model.outSchema.LatentSpace])*lforward22Pred.flatten(),
                                                                torch.tensor(0.0))

                losses[1] += lForwardThirdMom.data.item() / self.wLatent
                lForward += lForwardThirdMom

                # compute 4th moment loss on z
                lForwardFourthMom = self.wLatent * self.loss_fit(torch.mean(out[:, self.model.outSchema.LatentSpace])**4 + 6*(torch.mean(out[:, self.model.outSchema.LatentSpace])**2)*lforward22Pred.flatten() + (3*lforward22Pred.flatten()**2),
                                                                (3.0 * torch.eye((out[:, self.model.outSchema.LatentSpace].shape[1]),device=self.dev)).flatten())

                losses[1] += lForwardFourthMom.data.item() / self.wLatent
                lForward += lForwardFourthMom

                # compute 5th moment loss on z
                lForwardFifthMom = self.wLatent * self.loss_fit(torch.mean(out[:, self.model.outSchema.LatentSpace])**5 + 10*(torch.mean(out[:, self.model.outSchema.LatentSpace])**3)*lforward22Pred.flatten() + (15*torch.mean(out[:, self.model.outSchema.LatentSpace])*lforward22Pred.flatten()**2),
                                                                (0.0 * torch.eye((out[:, self.model.outSchema.LatentSpace].shape[1]),device=self.dev)).flatten())

                losses[1] += lForwardFifthMom.data.item() / self.wLatent
                lForward += lForwardFifthMom

            else:
                # compute forward MMD on z data
                outLatentGradOnly = torch.cat((out[:, self.model.outSchema.timeseries].data, 
                                               out[:, self.model.outSchema.LatentSpace]), 
                                              dim=1)
                unpaddedTarget = torch.cat((yzp[:, self.model.outSchema.timeseries], 
                                            yzp[:, self.model.outSchema.LatentSpace]), 
                                           dim=1)
            
                lForward2 = loss_factor_fwd_mmd_z * self.wLatent * self.loss_latent(out[:, self.model.outSchema.LatentSpace], yzp[:, self.model.outSchema.LatentSpace])
                losses[1] += lForward2.data.item() / self.wLatent
                lForward += lForward2
            
            lTot += lForward.data.item()

            lForward.backward()

            yzpRev = self.model.outSchema.fill({'timeseries': yClean[:],
                                                'LatentSpace': out[:, self.model.outSchema.LatentSpace].data},
                                               zero_pad_fn=pad_fn)

            if extra_z:
                outRev = self.model(yzpRev, rev=True)
                #outRev = torch.cat((outRev[:, self.model.inSchema.amp[0]:self.model.inSchema.tau[-1]+1],
                #                        outRev[:, self.model.inSchema.yNoise]),
                #                       dim=1)
                outRevRand = self.model(yzpRevRand, rev=True)
                outRevRand = torch.cat((outRevRand[:, self.model.inSchema.amp[0]:self.model.inSchema.tau[-1]+1],
                                        outRevRand[:, self.model.inSchema.yNoise]),
                                       dim=1)                
            else:
                outRev = self.model(yzpRev, rev=True)
                outRevRand = self.model(yzpRevRand, rev=True)

            # THis guy should have been OUTREVRAND!!!
            # xBack = torch.cat((outRevRand[:, self.model.inSchema.ne],
            #                    outRevRand[:, self.model.inSchema.temperature],
            #                    outRevRand[:, self.model.inSchema.vel]),
            #                   dim=1)
            # lBackward = self.wRev * wRevScale * self.loss_backward(xBack, x.reshape(self.miniBatchSize, -1))
            if extra_z:
                #xzp_bMMD=torch.cat((xzp[:, self.model.inSchema.amp[0]:self.model.inSchema.tau[-1]+1],
                #                   xzp[:, self.model.inSchema.yNoise]), dim=1)
                #lBackward = self.wRev * wRevScale * self.loss_fit(outRev,
                #                                                  xzp_bMMD)
                kld_loss = torch.nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean')
                lBackward1 = loss_factor_rev_mse_x * self.wRev * kld_loss(outRev[:, self.model.inSchema.amp[0]:self.model.inSchema.tau[-1]+1],
                                                                  xzp[:, self.model.inSchema.amp[0]:self.model.inSchema.tau[-1]+1])
                lBackward2 = loss_factor_rev_mse_n * self.wRev * kld_loss(outRev[:, self.model.inSchema.yNoise],
                                                                  xzp[:, self.model.inSchema.yNoise])
                lBackward = lBackward1 + lBackward2

            else:
                lBackward = self.wRev * wRevScale * self.loss_backward(outRev[:, self.model.inSchema.amp[0]:self.model.inSchema.tau[-1]+1], 
                                                                       xp[:, self.model.inSchema.amp[0]:self.model.inSchema.tau[-1]+1])

            scale = wRevScale if wRevScale != 0 else 1.0
            losses[2] += (lBackward1.data.item() / (self.wRev * scale)) + (lBackward1.data.item() / (self.wRev * scale))
            #TODO: may need to uncomment this
            #lBackward2 += 0.5 * self.wPred * self.loss_fit(outRev, xp)
            if extra_z:
                #lBackward2 = 0.5 * self.wPred * self.loss_fit(outRev,
                #                                               xzp_bMMD)
                losses[3] += 0.0
                lTot += lBackward.data.item()
            else:
                lBackward2 = 0.5 * self.wPred * self.loss_fit(outRev[:, self.model.inSchema.amp[0]:self.model.inSchema.tau[-1]+1], 
                                                                  xp[:, self.model.inSchema.amp[0]:self.model.inSchema.tau[-1]+1])
                losses[3] += lBackward2.data.item() / self.wPred * 2
                lBackward += lBackward2
            
                lTot += lBackward.data.item()

            lBackward.backward()

            for p in self.model.parameters():
                p.grad.data.clamp_(-15.0, 15.0)

            self.optim.step()

        losses = [l / miniBatchIdx for l in losses]
        return lTot / miniBatchIdx, losses

    def test(self, mcmc_post, maxBatches=10, extra_z=False):
        self.model.eval()

        # get ideal MMD/MSE backward pass
        #with torch.no_grad():
        #    for cnt in mcmc_post.shape[0]:
        #        loss_test = self.loss_backward(samples[cnt,:,k])
        #        print(loss_test)
        #        exit()

        forwardError = []
        backwardError = []

        batchIdx = 0
        
        if maxBatches == -1:
            maxBatches = len(self.atmosData.testLoader)

        pad_fn = lambda *x: torch.zeros(*x, device=self.dev) # 10 * torch.ones(*x, device=self.dev)
        randn = lambda *x: torch.randn(*x, device=self.dev)
        with torch.no_grad():
            for x, y in self.atmosData.testLoader:
                batchIdx += 1
                if batchIdx > maxBatches:
                    break

                x, y, y_sig = x.to(self.dev), y.to(self.dev), y_sig.to(self.dev)

                if extra_z:
                    inp = self.model.inSchema.fill({'amp': x[:, 0],
                                                't0': x[:, 1],
                                                'tau': x[:, 2],
                                                'yNoise': (y[:] - y_sig[:])},
                                               zero_pad_fn=pad_fn)
                else:
                    inp = self.model.inSchema.fill({'amp': x[:, 0],
                                                    't0': x[:, 1],
                                                    'tau': x[:, 2]},
                                                   zero_pad_fn=pad_fn)
                inpBack = self.model.outSchema.fill({'timeseries': y[:],
                                                     'LatentSpace': randn},
                                                    zero_pad_fn=pad_fn)
                                                    
                out = self.model(inp)
                f = self.loss_fit(out[:, self.model.outSchema.timeseries], y[:])
                forwardError.append(f)

                outBack = self.model(inpBack, rev=True)
#                 b = self.loss_fit(out[:, self.model.inSchema.ne], x[:, 0]) + \
#                     self.loss_fit(out[:, self.model.inSchema.temperature], x[:, 1]) + \
#                     self.loss_fit(out[:, self.model.inSchema.vel], x[:, 2])
                b = self.loss_backward(outBack, inp)
                backwardError.append(b)
        
            fE = torch.mean(torch.tensor(forwardError))
            bE = torch.mean(torch.tensor(backwardError))

            return fE, bE, out, outBack
        
    def review_mmd(self):
        with torch.no_grad():
            # Latent MMD
            loadIter = iter(self.atmosData.testLoader)
            # This is fine and doesn't load the first batch in testLoader every time, as shuffle=True
            x1, y1 = next(loadIter)
            x1, y1 = x1.to(self.dev), y1.to(self.dev)
            pad_fn = lambda *x: torch.zeros(*x, device=self.dev) # 10 * torch.ones(*x, device=self.dev)
            randn = lambda *x: torch.randn(*x, device=self.dev)
            xp = self.model.inSchema.fill({'amp': x1[:, 0],
                                           't0': x1[:, 1],
                                           'tau': x1[:, 2]},
                                          zero_pad_fn=pad_fn)
            yp = self.model.outSchema.fill({'timeseries': y1[:], 
                                           'LatentSpace': randn},
                                          zero_pad_fn=pad_fn)
            yFor = self.model(xp)
            yForNp = torch.cat((yFor[:, self.model.outSchema.timeseries], yFor[:, self.model.outSchema.LatentSpace]), dim=1).to(self.dev)
            ynp = torch.cat((yp[:, self.model.outSchema.timeseries], yp[:, self.model.outSchema.LatentSpace]), dim=1).to(self.dev)

            # Backward MMD
            xBack = self.model(yp, rev=True)

            r = np.logspace(np.log10(0.5), np.log10(500), num=2000)
            mmdValsFor = []
            mmdValsBack = []
            if self.mmFns is None:
                self.mmFns = []
                for a in r:
                    mm = mmd_multiscale_on(self.dev, alphas=[float(a)])
                    self.mmFns.append(mm)
                    
            for mm in self.mmFns:
                mmdValsFor.append(mm(yForNp, ynp).item())
                mmdValsBack.append(mm(xp[:, self.model.inSchema.amp[0]:self.model.inSchema.tau[-1]+1], xBack[:, self.model.inSchema.amp[0]:self.model.inSchema.tau[-1]+1]).item())


            def find_new_mmd_idx(a):
                aRev = a[::-1]
                for i, v in enumerate(a[-2::-1]):
                    if v < aRev[i]:
                        return min(len(a)-i, len(a)-1)
            mmdValsFor = np.array(mmdValsFor)
            mmdValsBack = np.array(mmdValsBack)
            idxFor = find_new_mmd_idx(mmdValsFor)
            idxBack = find_new_mmd_idx(mmdValsBack)
#             idxFor = np.searchsorted(r, 2.0) if idxFor is None else idxFor
#             idxBack = np.searchsorted(r, 2.0) if idxBack is None else idxBack
            idxFor = idxFor if not idxFor is None else np.searchsorted(r, 2.0)
            idxBack = idxBack if not idxBack is None else np.searchsorted(r, 2.0)

            self.loss_backward = mmd_multiscale_on(self.dev, alphas=[float(r[idxBack])])
            self.loss_latent = mmd_multiscale_on(self.dev, alphas=[float(r[idxFor])])

            return r, mmdValsFor, mmdValsBack, idxFor, idxBack


class AtmosData:
    def __init__(self, dataLocations, test_split, resampleWl=None):
        if type(dataLocations) is str:
            dataLocations = [dataLocations]

        #with open(dataLocations[0], 'rb') as p:
        #    data = pickle.load(p)
        data={'pos': h5py.File(dataLocations[0], 'r')['pos'][:],
              'labels': h5py.File(dataLocations[0], 'r')['labels'][:],
              'x': h5py.File(dataLocations[0], 'r')['x'][:],
              'sig': h5py.File(dataLocations[0], 'r')['sig'][:]}

        if len(dataLocations) > 1:
            for dataLocation in dataLocations[1:]:
                with open(dataLocation, 'rb') as p:
                    d = pickle.load(p)

                for k in data.keys():
                    if k == 'wavelength' or k == 'z' or k == 'lineInfo':
                        continue
                    if k == 'line':
                        for i in range(len(data['line'])):
                            data[k][i] += d[k][i]
                    else:
                        try:
                            data[k] += d[k]
                        except KeyError:
                            pass

        #TODO: may need to not log the training data
        self.pos = data['pos'][:]
        self.labels = data['labels'][:]
        self.sig = data['sig'][:]
        self.pos_test = data['pos'][-test_split:]
        self.labels_test = data['labels'][-test_split:]
        self.sig_test = data['sig'][-test_split:]
        self.x = data['x']
        data['pos']=data['pos'][:-test_split]
        data['labels']=data['labels'][:-test_split]
        self.amp = torch.tensor(data['pos'][0]).float()#.log10_()
        self.t0 = torch.tensor(data['pos'][1]).float()#.log10_()
        self.tau = torch.tensor(data['pos'][2]).float()#.log10()
        self.timeseries = torch.tensor(data['labels'][:]).float()#.log10()
        self.atmosIn=data['pos'][:]
        self.atmosOut=data['labels'][:]
        self.atmosSig=data['sig'][:-test_split]

    def split_data_and_init_loaders(self, batchSize, splitSeed=41, padLines=False, linePadValue='Edge', zeroPadding=0, testingFraction=0.2):
        self.batchSize = batchSize

        #if padLines and linePadValue == 'Edge':
        #    lPad0Size = (self.ne.shape[1] - self.lines[0].shape[1]) // 2
        #    rPad0Size = self.ne.shape[1] - self.lines[0].shape[1] - lPad0Size
        #    lPad1Size = (self.ne.shape[1] - self.lines[1].shape[1]) // 2
        #    rPad1Size = self.ne.shape[1] - self.lines[1].shape[1] - lPad1Size
        #    if any(np.array([lPad0Size, rPad0Size, lPad1Size, rPad1Size]) <= 0):
        #        raise ValueError('Cannot pad lines as they are already bigger than/same size as the profiles!')
        #    lPad0 = torch.ones(self.lines[0].shape[0], lPad0Size) * self.lines[0][:, 0].unsqueeze(1)
        #    rPad0 = torch.ones(self.lines[0].shape[0], rPad0Size) * self.lines[0][:, -1].unsqueeze(1)
        #    lPad1 = torch.ones(self.lines[1].shape[0], lPad1Size) * self.lines[1][:, 0].unsqueeze(1)
        #    rPad1 = torch.ones(self.lines[1].shape[0], rPad1Size) * self.lines[1][:, -1].unsqueeze(1)

        #    self.lineOut = torch.stack([torch.cat((lPad0, self.lines[0], rPad0), dim=1), torch.cat((lPad1, self.lines[1], rPad1), dim=1)]).permute(1, 0, 2)
        #elif padLines:
        #    lPad0Size = (self.ne.shape[1] - self.lines[0].shape[1]) // 2
        #    rPad0Size = self.ne.shape[1] - self.lines[0].shape[1] - lPad0Size
        #    lPad1Size = (self.ne.shape[1] - self.lines[1].shape[1]) // 2
        #    rPad1Size = self.ne.shape[1] - self.lines[1].shape[1] - lPad1Size
        #    if any(np.array([lPad0Size, rPad0Size, lPad1Size, rPad1Size]) <= 0):
        #        raise ValueError('Cannot pad lines as they are already bigger than/same size as the profiles!')
        #    lPad0 = torch.ones(self.lines[0].shape[0], lPad0Size) * linePadValue
        #    rPad0 = torch.ones(self.lines[0].shape[0], rPad0Size) * linePadValue
        #    lPad1 = torch.ones(self.lines[1].shape[0], lPad1Size) * linePadValue
        #    rPad1 = torch.ones(self.lines[1].shape[0], rPad1Size) * linePadValue

        #    self.lineOut = torch.stack([torch.cat((lPad0, self.lines[0], rPad0), dim=1), torch.cat((lPad1, self.lines[1], rPad1), dim=1)]).permute(1, 0, 2)
        #else:
        #    self.lineOut = torch.stack([self.lines[0], self.lines[1]]).permute(1, 0, 2)

        #indices = np.arange(self.atmosIn.shape[0])
        #np.random.RandomState(seed=splitSeed).shuffle(indices)

        # split off 20% for testing
        #maxIdx = int(self.atmosIn.shape[0] * (1.0 - testingFraction)) + 1
        #if zeroPadding != 0:
        #    trainIn = torch.cat((self.atmosIn[indices][:maxIdx], torch.zeros(maxIdx, self.atmosIn.shape[1], zeroPadding)), dim=2)
        #    trainOut = torch.cat((self.lineOut[indices][:maxIdx], torch.zeros(maxIdx, self.lineOut.shape[1], zeroPadding)), dim=2)
        #    testIn = torch.cat((self.atmosIn[indices][maxIdx:], torch.zeros(self.atmosIn.shape[0] - maxIdx, self.atmosIn.shape[1], zeroPadding)), dim=2)
        #    testOut = torch.cat((self.lineOut[indices][maxIdx:], torch.zeros(self.atmosIn.shape[0] - maxIdx, self.lineOut.shape[1], zeroPadding)), dim=2)
        #else:
        test_num = int(self.atmosIn.shape[0]*testingFraction)

        trainIn = self.atmosIn[:-test_num]
        trainOut = self.atmosOut[:-test_num]
        trainSig = self.atmosSig[:-test_num]
        testIn = self.atmosIn[-test_num:]
        testOut = self.atmosOut[-test_num:]
        testSig = self.atmosSig[-test_num:]

        self.testLoader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(torch.tensor(testIn), torch.tensor(testOut), torch.tensor(testSig)), 
                    batch_size=batchSize, shuffle=True, drop_last=True)
        self.trainLoader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(torch.tensor(trainIn), torch.tensor(trainOut), torch.tensor(trainSig)),
                    batch_size=self.batchSize, shuffle=True, drop_last=True)
                    

