import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import six
import collections
import copy
from scipy.stats import norm
from itertools import product
from sklearn.metrics import mean_squared_error
from random import sample 
from sklearn.datasets import make_moons
from collections import defaultdict
SKIP_TYPES = six.string_types



class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, tensor_names, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param tensor_names: name of tensors (for feed_dict)
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.tensor_names = tensor_names

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = {}
        for k in range(len(self.tensor_names)):
            batch.update({self.tensor_names[k]: self.tensors[k][self.i:self.i+self.batch_size]})
        self.i += self.batch_size
        return batch
        

    def __len__(self):
        return self.n_batches

class Meters:
    def __init__(self):
        self.data = defaultdict(list)

    def update(self, updates=None, value=None, n=1, **kwargs):
        """
        Example:
            >>> meters.update(key, value)
            >>> meters.update({key1: value1, key2: value2})
            >>> meters.update(key1=value1, key2=value2)
        """
        if updates is None:
            updates = {}
        if updates is not None and value is not None:
            updates = {updates: value}
        updates.update(kwargs)
        for k, v in updates.items():
            self.data[k].append(v)

    @property
    def avg(self):
        averages = {}
        for key, values in self.data.items():
            averages[key] = sum(values) / len(values)
        return averages

    def get_values(self, key):
        return self.data[key]

def get_activation(act):
    if isinstance(act, nn.Module):
        return act
    assert type(act) is str, 'Unknown type of activation: {}.'.format(act)
    act_lower = act.lower()
    if act_lower == 'relu':
        return nn.ReLU(True)
    elif act_lower == 'selu':
        return nn.SELU(True)
    elif act_lower == 'sigmoid':
        return nn.Sigmoid()
    elif act_lower == 'tanh':
        return nn.Tanh()
    else:
        try:
            return getattr(nn, act)
        except AttributeError:
            raise ValueError('Unknown activation function: {}.'.format(act))


def get_optimizer(optimizer, model, *args, **kwargs):
    if isinstance(optimizer, (optim.Optimizer)):
        return optimizer

    if type(optimizer) is str:
        try:
            optimizer = getattr(optim, optimizer)
        except AttributeError:
            raise ValueError('Unknown optimizer type: {}.'.format(optimizer))
    return optimizer(filter(lambda p: p.requires_grad, model.parameters()), *args, **kwargs)
    

def stmap(func, iterable):
    if isinstance(iterable, six.string_types):
        return func(iterable)
    elif isinstance(iterable, (collections.Sequence, collections.UserList)):
        return [stmap(func, v) for v in iterable]
    elif isinstance(iterable, collections.Set):
        return {stmap(func, v) for v in iterable}
    elif isinstance(iterable, (collections.Mapping, collections.UserDict)):
        return {k: stmap(func, v) for k, v in iterable.items()}
    else:
        return func(iterable)




def _as_numpy(o):
    from torch.autograd import Variable
    if isinstance(o, SKIP_TYPES):
        return o
    if isinstance(o, Variable):
        o = o
    if torch.is_tensor(o):
        return o.cpu().numpy()
    return np.array(o)


def as_numpy(obj):
    return stmap(_as_numpy, obj)


def _as_float(o):
    if isinstance(o, SKIP_TYPES):
        return o
    if torch.is_tensor(o):
        return o.item()
    arr = as_numpy(o)
    assert arr.size == 1
    return float(arr)


def as_float(obj):
    return stmap(_as_float, obj)




# Create a simple dataset
def create_twomoon_dataset(n, p):
    relevant, y = make_moons(n_samples=n, shuffle=True, noise=0.1, random_state=None)
    noise_vector = norm.rvs(loc=0, scale=1, size=[n,p-2])
    X = np.concatenate([relevant, noise_vector], axis=1)
    return dict(X=X, y=y.flatten())



def get_model_size(model_path):
    size = []
    for j, _model in enumerate(model_path):
        if hasattr(_model.mlp[0], "__getitem__"):
            weight = _model.mlp[0][0].weight
        else:
            weight = _model.mlp[0].weight
        if torch.isnan(weight).any():
            size.append(np.nan)
        else:
            var_sel= torch.sum(torch.abs(weight), 0)>1e-6
            size.append(int(sum(var_sel)))
    return np.array(size)


def get_param_combination(param_grid):
    param_list = [param_grid[key] for key in param_grid]
    comb_list = []
    for comb in product(*param_list):
        comb_dict = {key: value for key, value in zip(param_grid.keys(), comb)}
        comb_list.append(comb_dict)
    return comb_list
         



def get_mse(model, X, y, best=True):
    if best:
        model._model = model.best_model
    if model.drop_input:
        X = X[:,model._index_path[model.best_lam_ind]]
    y_hat= model.predict(X)
    return mean_squared_error(y, y_hat)


def nanargmax_safe(arr, axis=None):
    if np.isnan(arr).all():
        return len(arr.flatten())-1
    else:
        return np.nanargmax(arr, axis=axis)
    

def _standard_truncnorm_sample(lower_bound, upper_bound, sample_shape=torch.Size()):
    r"""
    Implements accept-reject algorithm for doubly truncated standard normal distribution.
    (Section 2.2. Two-sided truncated normal distribution in [1])
    [1] Robert, Christian P. "Simulation of truncated normal variables." Statistics and computing 5.2 (1995): 121-125.
    Available online: https://arxiv.org/abs/0907.4010
    Args:
        lower_bound (Tensor): lower bound for standard normal distribution. Best to keep it greater than -4.0 for
        stable results
        upper_bound (Tensor): upper bound for standard normal distribution. Best to keep it smaller than 4.0 for
        stable results
    """
    x = torch.randn(sample_shape)
    done = torch.zeros(sample_shape).byte() 
    while not done.all():
        proposed_x = lower_bound + torch.rand(sample_shape) * (upper_bound - lower_bound)
        if (upper_bound * lower_bound).lt(0.0):  # of opposite sign
            log_prob_accept = -0.5 * proposed_x**2
        elif upper_bound < 0.0:  # both negative
            log_prob_accept = 0.5 * (upper_bound**2 - proposed_x**2)
        else:  # both positive
            assert(lower_bound.gt(0.0))
            log_prob_accept = 0.5 * (lower_bound**2 - proposed_x**2)
        prob_accept = torch.exp(log_prob_accept).clamp_(0.0, 1.0)
        accept = torch.bernoulli(prob_accept).byte() & ~done
        if accept.any():
            accept = accept.bool()
            x[accept] = proposed_x[accept]
            accept = accept.byte()
            done |= accept
    return x

def simulate_data(n_samples, n_features, scale=2, shape=2, data_type="cox", censoring_rate=0.1,
 times_distribution="weibull"):
    """  n_samples=10
        n_features=5
        scale=0.5
        shape=0.5
        censoring_factor=2
        times_distribution="weibull" """  
    mu=np.zeros(n_features)
    cov=np.diag(np.ones(n_features))
    X=np.random.multivariate_normal(mean=mu, cov=cov, size=n_samples)
    u= X[:,0] + X[:,1]* X[:,0]+ np.log(np.abs(X[:,1])+0.1)+ np.exp(X[:,2]+X[:,3])  
         
    if data_type == "cox":
        # Simulation of true times
        E = np.random.exponential(scale=1., size=n_samples)
        E *= np.exp(-u)
        if times_distribution == "weibull":
            T = (E / scale ) ** (1. / shape)
        else:
            # There is not point in this test, but let's do it like that
            # since we're likely to implement other distributions
            T = (E / scale ) ** (1. / shape)
        censor_index = sample(range(n_samples), int(n_samples * censoring_rate))
        t=copy.deepcopy(T) 
        t[censor_index] = np.random.uniform(0, T[censor_index])
        e=(T==t)
        return dict(X=X, T=t, E=e)
    elif data_type == "regression":
        eps = np.random.normal(loc=0, scale=1, size=n_samples)
        y = u + eps
        return dict(X=X,y=y)
    elif data_type == "classification":
        p = 1/(1+np.exp(-u))
        y = np.random.binomial(1,p=p)
        return dict(X=X, y=y.flatten())