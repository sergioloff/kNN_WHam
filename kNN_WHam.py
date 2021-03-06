import datetime
import math
import numpy as np
import numpy.ma as ma
from matplotlib import pyplot as plt
import os
import torch
from torch import nn
import cupy as cp
from utils.WHDxCacl import WHDxFunction
from pytorch_lightning import seed_everything
from utils.LoadMNIST import LoadMNIST_0Lab as lm


class kNNFit(nn.Module):
    def __init__(self, D, eps, init_W=None):
        super(kNNFit, self).__init__()
        if (init_W is None):
            init_W = np.full([D], math.sqrt(1/D), dtype=np.float32)
        else:
            init_W = init_W.astype(np.float32)
        self.WHDxFunc = WHDxFunction.apply
        self.t_W = nn.Parameter(torch.from_numpy(init_W))
        self.alpha = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.k = nn.Parameter(torch.tensor(0, dtype=torch.float32)) # horiz shift left/right [-1,1]
        self.j = nn.Parameter(torch.tensor(0, dtype=torch.float32)) # curvature  [-1,1]
        self.eps = eps

    def forward(self, x):
        t_X1_T_bits, t_X2_T_bits, t_X2_byDimBlock_T_bits, t_Y_knn = x
        t_WHDx = self.WHDxFunc(t_X1_T_bits, t_X2_T_bits, t_X2_byDimBlock_T_bits, self.t_W) # ((x1 - x2) * W).abs()
        t_norm_W = torch.linalg.norm(self.t_W, 1) + self.eps
        t_FQx = self.F(1 - t_WHDx / t_norm_W) 
        t_score = (t_FQx * torch.unsqueeze(t_Y_knn, 1)).mean(0) 
        t_shiftedScore = self.beta * (t_score + self.alpha)
        return t_shiftedScore

    def F(self, x): 
        x = torch.clamp(x, 0, 1)
        j = torch.clamp(self.j, -1, 1)
        k = torch.clamp(self.k, -1, 1)
        inner = x * (1 - k) / (k * (1 - 2 * abs(x)) + 1)
        outer = (1 + (inner * 2 - 1) * (1 + j) / (- j * (1 - 2 * abs(inner * 2 - 1)) + 1)) / 2
        return outer 


def GenerateEpochIxs(M):
    return torch.randperm(M)

def GenerateBatchIxs(epochIxs, batchLength, batchIx):
    M = epochIxs.shape[0]
    t_holdoutIxs = epochIxs[batchIx * batchLength:(batchIx + 1) * batchLength]
    t_knnIxs = torch.cat([epochIxs[:batchIx * batchLength], epochIxs[(batchIx + 1) * batchLength:]])
    return t_holdoutIxs, t_knnIxs 


def GenerateBatchData(t_X_bits_T, t_Y, t_Y_01, t_holdoutIxs, t_knnIxs):
    t_Y_knn = t_Y[t_knnIxs]
    t_Y_holdout_01 = t_Y_01[t_holdoutIxs]
    t_X_bits_T_knn = t_X_bits_T[:,t_knnIxs].contiguous()
    t_X_bits_T_holdout = t_X_bits_T[:,t_holdoutIxs].contiguous()
    t_X_bits_T_holdout_byDimBlock4 = WHDxFunction.GetDimBlock4(t_X_bits_T_holdout)
    return t_X_bits_T_knn, t_Y_knn, t_X_bits_T_holdout, t_X_bits_T_holdout_byDimBlock4, t_Y_holdout_01


def formatDateTime(dtn=None):
    if dtn is None:
        dtn = datetime.datetime.now()
    return '{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'.format(dtn.year, dtn.month, dtn.day, dtn.hour,dtn.minute,dtn.second)


if not torch.cuda.is_available():
    raise Exception('cpu implementation not available')

deviceIx = 0 
torch_device = torch.device('cuda:' + str(deviceIx))
cp_device = cp.cuda.Device(deviceIx)
cp_device.use()

seed_everything(seed = 0)
eps = 1e-6
opt_lr = 0.001
l1_lambda = 0.0
l2_lambda = 0.1
batchLength = 64
epochs = 3
MINST_th = 230
dataFolder = 'dat' 
X, Y_01 = lm(os.path.join(dataFolder, 'mnist-train-images.npy'), os.path.join(dataFolder, 'mnist-train-labels.npy'), th = MINST_th)
X_test, Y_01_test = lm(os.path.join(dataFolder, 'mnist-val-images.npy'), os.path.join(dataFolder, 'mnist-val-labels.npy'), th = MINST_th)
D = X.shape[1] * X.shape[2]
totBatches = X.shape[0] // batchLength
M = totBatches * batchLength
N = (X_test.shape[0] // 64) * 64


print('tot attribs={}\r\ntot train samples={}\r\ntot test samples={}\r\ntot batches={}\r\nbatchLength={}\r\ntot epochs={}\r\nopt_lr={}\r\nl1_lambda={}\r\nl2_lambda={}\r\nMINST_th={}'
    .format(D, M, N, totBatches, batchLength, epochs, opt_lr, l1_lambda, l2_lambda, MINST_th))

if (M % batchLength != 0):
    raise Exception('train set size must be divisible by batch length')

loss_func = nn.BCEWithLogitsLoss(reduction='mean')
model = kNNFit(D, eps).to(torch_device)
opt = torch.optim.Adam(model.parameters(), lr=opt_lr, eps=eps)

# trim data to pow2-boundaries, etc
X = X[:M]
Y_01 = Y_01[:M]
Y = np.where(Y_01 > 0.5, 1, -1).astype(np.float32)
X_test = X_test[:N]
Y_01_test = Y_01_test[:N]
Y_test = np.where(Y_01_test > 0.5, 1, -1).astype(np.float32)
# load data onto the gpu
t_X_bits_T = torch.tensor(np.packbits(X.reshape([X.shape[0],-1]).astype(np.int32), axis=1).view(np.int32), device=torch_device).T.contiguous()
t_Y = torch.tensor(Y, device=torch_device).contiguous()
t_Y_01 = torch.tensor(np.where(Y >= 0, 1.0, 0.0).astype(np.float32), device=torch_device).contiguous()
t_X_bits_T_test = torch.tensor(np.packbits(X_test.reshape([X_test.shape[0],-1]).astype(np.int32), axis=1).view(np.int32), device=torch_device).T.contiguous()
t_X_bits_T_byDimBlock4_test = WHDxFunction.GetDimBlock4(t_X_bits_T_test)
t_Y_test = torch.tensor(Y_test, device=torch_device).contiguous()
t_Y_01_test = torch.tensor(Y_01_test, device=torch_device).contiguous()

losses = []
accs = []
losses_test = []
accs_test = []

for epoch in range(epochs):
    # a full epoch is the partitioning of the whole training datum indexes, shuffled, into 'totBatches' batches
    epochIxs = GenerateEpochIxs(M)
    for batch in range(totBatches):
        with torch.no_grad():
            # each batch consists of the sampling of 'batchLength' elements, without 
            # replacement, from the indexes of the training set
            # 't_holdoutIxs' are the set of indexes of the training data that will search for their nearest neighbours
            # 't_knnIxs' are the set of indexes of the training data that will be used as knn neighbors
            t_holdoutIxs, t_knnIxs = GenerateBatchIxs(epochIxs, batchLength, batch)
            # get subsets of the training data on the gpu, based on the current batch's indexes
            t_X_bits_T_knn, t_Y_knn, t_X_bits_T_holdout, t_X_bits_T_holdout_byDimBlock4, t_Y_holdout_01 = \
                GenerateBatchData(t_X_bits_T, t_Y, t_Y_01, t_holdoutIxs, t_knnIxs)

        t_predict_y = model([t_X_bits_T_knn, t_X_bits_T_holdout, t_X_bits_T_holdout_byDimBlock4, t_Y_knn])
        
        loss = loss_func(t_predict_y, t_Y_holdout_01) + \
            l1_lambda * torch.linalg.norm(model.t_W, 1) +\
            l2_lambda * torch.linalg.norm(model.t_W, 2)

        loss.backward()
        opt.step()
        opt.zero_grad()

        def CalcAcc(t_y_pred, t_y_true01):
            t_y_pred01 = torch.where(t_y_pred > 0, 1, 0)
            acc = (100.0 * (1.0 - abs(t_y_pred01 - t_y_true01).sum() / t_y_pred01.shape[0])).item()
            return acc

        with torch.no_grad():
            # valid/test loss+acc. move to epoch end if performance is an issue
            t_predict_y_test = model([t_X_bits_T, t_X_bits_T_test, t_X_bits_T_byDimBlock4_test, t_Y])
            loss_test = loss_func(t_predict_y_test, t_Y_01_test).item()
            acc_test = CalcAcc(t_predict_y_test, t_Y_01_test)

            acc = CalcAcc(t_predict_y, t_Y_holdout_01)
            loss = loss.item()
            losses.append(loss)
            losses_test.append(loss_test)
            accs.append(acc)
            accs_test.append(acc_test)
            print('{} epoch={:03d}, batch={:04d}, loss={:6.6f}, acc={:3.2f}%, test loss={:6.6f}, test acc={:3.2f}% '
                  .format(formatDateTime(), epoch, batch, loss, acc, loss_test, acc_test))


plt.imshow(abs(model.t_W.detach()).reshape([32,32]).detach().cpu().numpy(), interpolation='none')
plt.title('trained W at acc={:3.2f}% (test acc={:3.2f}%)'.format(acc, acc_test))
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
lns1 = ax.plot(losses, color='black', label='train loss')
lns2 = ax.plot(losses_test, color='grey', label='test loss')
lns3 = ax2.plot(accs, color='red', label='train acc')
lns4 = ax2.plot(accs_test, color='pink', label='test acc')
lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='center right')
ax.grid()
ax.set_xlabel('iteration')
ax.set_ylabel('loss')
ax2.set_ylabel('acc%')
ax2.set_ylim(0, 100)
plt.show()

