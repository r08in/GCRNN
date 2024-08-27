import torch.nn as nn
import torch
import numpy as np

from gcrnet.mlp import MLPModel
from gcrnet.losses import PartialLogLikelihood
from gcrnet.penalty import *




class ConcaveRegularMLPModel(MLPModel):
    def __init__(self, input_dim, output_dim, hidden_dims,  activation='relu',
                 outer_penalty=None, inner_penalty=None, lam=0, alpha=0, gamma=1, output_bias=False):
        super().__init__(input_dim, output_dim, hidden_dims, activation=activation, output_bias=output_bias)
        self.outer_penalty = outer_penalty
        self.inner_penalty = inner_penalty
        self.lam=lam
        self.alpha=alpha
        self.gamma=gamma
        if outer_penalty is not None and inner_penalty is not None:
            assert type(outer_penalty) is str, 'Unknown type of outer penalty: {}.'.format(outer_penalty)
            assert type(inner_penalty) is str, 'Unknown type of inner penalty: {}.'.format(inner_penalty)

    def concave_soft_thresh_update(self):
        if self.outer_penalty is None or self.inner_penalty is None:
            return 0 
        len_sparse_layer=1
        penalty_val = 0
        with torch.no_grad():
            for i in range(len_sparse_layer):
                if hasattr(self.mlp[i], "__getitem__"):
                    for j in range(self.mlp[i][0].weight.shape[1]):
                        self.mlp[i][0].weight.data[:,j]=ConcaveSoftThresh(self.mlp[i][0].weight[:,j], lam=self.lam, gamma=self.gamma, 
                                                            outer_penalty=self.outer_penalty, inner_penalty=self.inner_penalty)
                        penalty_val = penalty_val +get_penalty_val(self.mlp[i][0].weight.data[:,j], lam=self.lam, 
                                                            outer_penalty=self.outer_penalty, inner_penalty=self.inner_penalty)
                        for a in self.mlp[i][0].weight.data[:,j]:
                            if torch.isnan(a).any():
                                raise ValueError("NaNs detected in inputs, please correct or drop.")
                else:
                    for j in range(self.mlp[i].weight.shape[1]):
                        self.mlp[i].weight.data[:,j]=ConcaveSoftThresh(self.mlp[i].weight[:,j], lam=self.lam, gamma=self.gamma,
                                                            outer_penalty=self.outer_penalty, inner_penalty=self.inner_penalty)
                        val = get_penalty_val(self.mlp[i].weight.data[:,j], lam=self.lam, 
                                                            outer_penalty=self.outer_penalty, inner_penalty=self.inner_penalty)                 
                        penalty_val = penalty_val + val
        return self.gamma*penalty_val 
                
    def get_concave_penalty_val(self):
        if self.outer_penalty is None or self.inner_penalty is None:
            return 0 
        len_sparse_layer=1
        penalty_val = 0
        with torch.no_grad():
            for i in range(len_sparse_layer):
                if hasattr(self.mlp[i], "__getitem__"):
                    for j in range(self.mlp[i][0].weight.shape[1]):
                        penalty_val = penalty_val +get_penalty_val(self.mlp[i][0].weight.data[:,j], lam=self.lam, 
                                                            outer_penalty=self.outer_penalty, inner_penalty=self.inner_penalty)
                else:
                    for j in range(self.mlp[i].weight.shape[1]):
                        val = get_penalty_val(self.mlp[i].weight.data[:,j], lam=self.lam, 
                                                            outer_penalty=self.outer_penalty, inner_penalty=self.inner_penalty)                 
                        penalty_val = penalty_val + val
        return self.gamma* penalty_val 
        

    def l2_regularize(self):
        if self.alpha != 0:
            return self.alpha * sum(p.pow(2.0).sum() for p in self.parameters())
        return 0

### Regression Model ###
class ConcaveRegularMLPRegressionModel(ConcaveRegularMLPModel):
    def __init__(self, input_dim, output_dim, hidden_dims,  activation='relu',
                    outer_penalty=None, inner_penalty=None, lam=0, alpha=0, gamma=1):
        super().__init__(input_dim, output_dim, hidden_dims, activation=activation,
         outer_penalty=outer_penalty, inner_penalty=inner_penalty, lam=lam, alpha=alpha, gamma=gamma, output_bias=True)
        self.loss = nn.MSELoss()

    def forward(self, feed_dict):
        pred = super().forward(feed_dict['X'])
        if pred.shape[1]==1:
            pred = pred.view(pred.size(0))
        if self.training:
            loss = self.loss(pred, feed_dict['y']) + self.l2_regularize()
            return loss, dict(), dict()
        else:
            return dict(pred=pred)

### Classification Model ###
class ConcaveRegularMLPClassificationModel(ConcaveRegularMLPModel):
    def __init__(self, input_dim, output_dim, hidden_dims, activation='relu',
                    outer_penalty=None, inner_penalty=None, lam=0, alpha=0, gamma=1):
        super().__init__(input_dim, output_dim, hidden_dims, activation=activation,
         outer_penalty=outer_penalty, inner_penalty=inner_penalty, lam=lam, alpha=alpha,  gamma=gamma, output_bias=True)
        # self.softmax = nn.Softmax(dim=1)
        # self.loss = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, feed_dict):
        logits = super().forward(feed_dict['X'])
        if logits.shape[1]==1:
            logits = logits.view(logits.size(0))
        if self.training:
            loss = self.loss(self.sigmoid(logits), feed_dict['y'].to(torch.float32) ) + self.l2_regularize()
            return loss, dict(), dict()
        else:
            return self._compose_output(logits)

    def _compose_output(self, logits):
        value = self.sigmoid(logits)
        pred = np.array(value > 0.5)*1
        return dict(prob=value, pred=pred, logits=logits)
### Cox Model
class ConcaveRegularMLPCoxModel(ConcaveRegularMLPModel):
    def __init__(self, input_dim, output_dim, hidden_dims,  activation='relu',
                    outer_penalty=None, inner_penalty=None, lam=0, alpha=0, gamma=1):
        super().__init__(input_dim, output_dim, hidden_dims, activation=activation,
         outer_penalty=outer_penalty, inner_penalty=inner_penalty, lam=lam, alpha=alpha, gamma=gamma,  output_bias=False)
        self.loss = PartialLogLikelihood 
        self.noties = 'noties'

    def forward(self, feed_dict):
        logits = super().forward(feed_dict['X'])
        if self.training:
            loss = self.loss(logits, feed_dict['E'].reshape(-1, 1), self.noties) + self.l2_regularize()
            return loss, logits, dict()
        else:
            return dict(logits=logits)

## Bivariate Cox Model
class ConcaveRegularMLPBivariateCoxModel(ConcaveRegularMLPModel):
    def __init__(self, input_dim, hidden_dims,  activation='relu',
                    outer_penalty=None, inner_penalty=None, lam=0, alpha=0, gamma=1):
        super().__init__(input_dim=input_dim, output_dim=2, hidden_dims=hidden_dims, activation=activation,
         outer_penalty=outer_penalty, inner_penalty=inner_penalty, lam=lam, alpha=alpha, gamma=gamma, output_bias=False)
        self.loss = PartialLogLikelihood 
        self.noties = 'noties'
        self.sort_id2 = None

    def forward(self, feed_dict):
        logits = super().forward(feed_dict['X'])
        if self.training:
            if self.sort_id2 is None:
                self.sort_id2 = torch.flip(torch.argsort(feed_dict['T2']), dims=[0])
            ordered_E2 = feed_dict['E2'][self.sort_id2]
            loss = self.loss(logits[:,0], feed_dict['E'].reshape(-1, 1), self.noties) + self.loss(logits[self.sort_id2,1], ordered_E2.reshape(-1, 1), self.noties) + self.l2_regularize()
            #loss = self.loss(logits[self.sort_id2,1], ordered_E2.reshape(-1, 1), self.noties) + self.l2_regularize()
            #loss = self.loss(logits[:,0], feed_dict['E'].reshape(-1, 1), self.noties) + self.l2_regularize()
            return loss, logits, dict()
        else:
            return dict(logits=logits)




    