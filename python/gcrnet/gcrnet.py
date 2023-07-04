import torch
import torch.nn as nn
import copy
from sklearn.metrics import r2_score
from gcrnet.models import  ConcaveRegularMLPCoxModel, ConcaveRegularMLPRegressionModel, ConcaveRegularMLPClassificationModel
from gcrnet.utils import Meters, get_optimizer, as_float, as_numpy, FastTensorDataLoader, nanargmax_safe, get_model_size, get_param_combination, _standard_truncnorm_sample
from gcrnet.losses import calc_concordance_index, PartialLogLikelihood
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


__all__ = ['gcrnet']



class GCRNet(BaseEstimator):
    def __init__(self, device, input_dim, output_dim=1, hidden_dims=[10,5], activation='relu', 
                penalty=None, lam=0, alpha=0,
                optimizer='Adam', learning_rate=0.001,  batch_size=100, weight_decay=0, 
                task_type='cox', drop_input=False, extra_args=None):
        """
        Constructs a GCRNet model.

        Parameters:
            device (torch.device): The device on which the model will be trained (e.g., torch.device("cpu") or torch.device("cuda")).
            input_dim (int): The input dimension of the model.
            output_dim (int): The output dimension of the model (default: 1).
            hidden_dims (list): The list of hidden layer dimensions (default: [10, 5]).
            activation (str): The activation function to use in the hidden layers (default: 'relu').
            penalty (None or str): The type of penalty to apply ('LASSO', 'MCP' or 'SCAD', default: None).
            lam (float or list): The regularization parameter(s) for concave regularization (default: 0).
            alpha (float): The alpha parameter for ridge regularization (default: 0).
            optimizer (str): The optimizer to use for training (default: 'Adam').
            learning_rate (float): The learning rate for the optimizer (default: 0.001).
            batch_size (int): The batch size for training (default: 100).
            weight_decay (float): The weight decay for the optimizer (default: 0).
            task_type (str): The type of task to perform ('cox', 'regression', or 'classification', default: 'cox').
            drop_input (bool): Whether to drop input features (dimension pruning) along the solution path based on dimen(default: False).
            extra_args (None or dict): Extra arguments to be passed to the model (default: None).
        """
        self.batch_size = batch_size
        self.activation = activation
        self.device = device
        self.task_type = task_type
        self.extra_args = extra_args
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.hidden_dims=hidden_dims
        self.activation=activation
        self.outer_penalty=penalty
        self.inner_penalty='l2'
        self.alpha = alpha
        self.optimizer=optimizer
        self.learning_rate=learning_rate
        self.tol = 1e-6
        self.weight_decay=weight_decay
        self.drop_input = drop_input
        if isinstance(lam, (int, float)):
            lam=[lam]
        self.lam=lam
        self._init_nn()
    
    def _init_nn(self):
        self._model_path =[]
        if self.drop_input:
            self._index_path=[]
        self._model = self.build_model()
        self._model.apply(self.init_weights)
        self._model = self._model.to(self.device)
        self._optimizer = get_optimizer(self.optimizer, self._model, lr=self.learning_rate, weight_decay=self.weight_decay)
        return self
    

    def get_device(self, device):
        if device == "cpu":
            device = torch.device("cpu")
        elif device == "cuda":
            args_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if args_cuda else "cpu")
        else:
            raise NotImplementedError("Only 'cpu' or 'cuda' is a valid option.")
        return device
        
        
    def init_weights(self, m, add=False):
        if isinstance(m, nn.Linear):
            shape = m.weight.shape
            stddev = torch.tensor(0.1)
            m.weight = nn.Parameter(_standard_truncnorm_sample(lower_bound=-2*stddev, upper_bound=2*stddev, 
                                sample_shape=shape))
            if m.bias!=None:
                torch.nn.init.zeros_(m.bias)


    def build_model(self):
        if self.task_type == 'classification':
            # self.metric = nn.CrossEntropyLoss()
            self.metric = nn.BCELoss()
            self.sigmoid = nn.Sigmoid()
            self.tensor_names = ('X','y')
            return ConcaveRegularMLPClassificationModel(self.input_dim, self.output_dim, self.hidden_dims, activation=self.activation,
                    outer_penalty=self.outer_penalty, inner_penalty=self.inner_penalty,
                      lam=self.lam[0], alpha=self.alpha)
                
        elif self.task_type == 'regression':
            self.metric = nn.MSELoss()
            self.tensor_names = ('X','y')
            return ConcaveRegularMLPRegressionModel(self.input_dim, self.output_dim, self.hidden_dims, activation=self.activation,
                    outer_penalty=self.outer_penalty, inner_penalty=self.inner_penalty, 
                      lam=self.lam[0], alpha=self.alpha)
                    
        elif self.task_type == 'cox':
            self.metric = PartialLogLikelihood
            self.tensor_names = ('X', 'E', 'T')
            return ConcaveRegularMLPCoxModel(self.input_dim, self.output_dim, self.hidden_dims, activation=self.activation,
                outer_penalty=self.outer_penalty, inner_penalty=self.inner_penalty, 
                  lam=self.lam[0], alpha=self.alpha)
        else:
            raise NotImplementedError()


    def get_dataloader(self, X, y, shuffle):
        if self.task_type == 'classification':
            data_loader = FastTensorDataLoader(torch.from_numpy(X).float().to(self.device), 
                        torch.from_numpy(y).long().to(self.device), tensor_names=self.tensor_names,
                        batch_size=self.batch_size, shuffle=shuffle)

        elif self.task_type == 'regression':
            data_loader = FastTensorDataLoader(torch.from_numpy(X).float().to(self.device), 
                        torch.from_numpy(y).float().to(self.device), tensor_names=self.tensor_names,
                        batch_size=self.batch_size, shuffle=shuffle)

        elif self.task_type == 'cox':
            assert isinstance(y, dict)
            data_loader = FastTensorDataLoader(torch.from_numpy(X).float().to(self.device), 
                        torch.from_numpy(y['E']).float().to(self.device),
                        torch.from_numpy(y['T']).float().to(self.device),
                        tensor_names=self.tensor_names,
                        batch_size=self.batch_size, shuffle=shuffle)

        else:
            raise NotImplementedError()

        return data_loader 

    def fit(self, X, y, init_num_epochs=None, num_epochs=200, verbose=True,  print_interval=1):
        if self.task_type =="cox":
            sort_idx = np.argsort(y['T'])[::-1]
            X=X[sort_idx]
            y={'E':np.array(y.iloc[sort_idx]['E']), 'T':np.array(y.iloc[sort_idx]['T'])}
        data_loader = self.get_dataloader(X, y, shuffle=False)
        self._init_nn()
        n_features = self.input_dim
        if self.drop_input:
            self._index_path =[]
            cur_index = np.arange(self.input_dim)

        for i, _lam in enumerate(self.lam):
            self._model.lam =  _lam
            self._optimizer = get_optimizer(self.optimizer, self._model, lr=self.learning_rate, weight_decay=self.weight_decay)
            n_epoch=num_epochs
            if i==0:
                n_epoch=init_num_epochs
                if init_num_epochs is None:
                    if len(self.lam)==1:
                        n_epoch = 5000
                    elif n_features > X.shape[0]: 
                        n_epoch=num_epochs
                    else:
                        n_epoch = 2000
            try:
                if n_features>0:
                    self.train(data_loader, n_epoch, verbose, print_interval)
                    weight = self._model.mlp[0][0].weight
                    var_sel= torch.sum(torch.abs(weight), 0)>1e-6
                    n_features = int(sum(var_sel))
                    if verbose:
                        print(f"Lambda{i}={_lam}, size={n_features}")
                if self.drop_input:
                    if n_features>0 and n_features<len(cur_index):
                        cur_index =  cur_index[var_sel]
                        weights = self._model.mlp[0][0].weight.data[:, var_sel]
                        biases =  self._model.mlp[0][0].bias.data
                        self._model.mlp[0][0]=nn.Linear(n_features, weights.shape[0], bias=True)
                        self._model.mlp[0][0].weight.data = weights
                        self._model.mlp[0][0].bias.data = biases
                        data_loader = self.get_dataloader(X[:,cur_index], y, shuffle=False)
                    self._index_path.append(cur_index) 
                self._model_path.append(copy.deepcopy(self._model))            
            except ValueError as e:
                print(e)
                self._model_path.append(copy.deepcopy(self._model))
                self._model.apply(self.init_weights)
                if self.drop_input:
                    self._index_path.append(cur_index)
                       
                
    def get_selection(self):
        w=self._model_path[self.best_lam_ind].mlp[0][0].weight
        var_sel= torch.sum(torch.abs(w), 0)>1e-6
        if self.drop_input:
            index = self._index_path[self.best_lam_ind]
            var_sel0 = np.zeros(self.input_dim, dtype=bool)
            var_sel0[index] = np.array(var_sel)
            var_sel = var_sel0            
        return var_sel


    def predict(self, X, key="logits", best=True):
        if best and hasattr(self, "best_model"): ## check if contain NAN
            self._model = self.best_model     
            if self.drop_input: 
                best_index = self._index_path[self.best_lam_ind]
                X = X[:, best_index]
        data_loader= FastTensorDataLoader(torch.from_numpy(X).float().to(self.device), tensor_names=self.tensor_names[0],
                        batch_size=X.shape[0], shuffle=False)
        res = []
        self._model.eval()
        for feed_dict in data_loader:
            with torch.no_grad():
                output_dict = self._model(feed_dict)
            output_dict_np = as_numpy(output_dict)
            res.append(output_dict_np[key])
        return np.concatenate(res, axis=0)


    def train(self, data_loader, nr_epochs=1000, verbose=True, print_interval=1):
        pre_loss = 1e8
        min_epochs = 0.5 * nr_epochs
        pre_weight = nn.utils.parameters_to_vector(self._model.parameters())
        self._model.train()
        penalty_val=self._model.get_concave_penalty_val()
        for epoch in range(1, 1 + nr_epochs):         
            for feed_dict in data_loader:
                pre_smooth_loss, logits, monitors = self._model(feed_dict) 
                loss =  as_float(pre_smooth_loss) + penalty_val
                self._optimizer.zero_grad()   
                pre_smooth_loss.backward()
                self._optimizer.step()
                for param in self._model.parameters():
                    if torch.isnan(param).any():
                        raise ValueError("NaNs detected in weights, use larger tuning parameter for l2 regularization.")
                penalty_val= self._model.concave_soft_thresh_update()
            if verbose and epoch % print_interval == 0:
                print(f'Epoch: {epoch}: loss={loss}')
            cur_weight = nn.utils.parameters_to_vector(self._model.parameters())
            wd =  as_float(torch.norm(pre_weight -  cur_weight, p=2))
            if min_epochs <= epoch and abs(pre_loss-loss)<self.tol and wd < self.tol:
                break
            else:
                pre_loss = loss
                pre_weight = cur_weight
    
    def validate_step(self, feed_dict, metric, meters=None, mode='valid'):
        with torch.no_grad():
            pred = self._model(feed_dict)
        if self.task_type == 'classification':
            result = metric(self.sigmoid(pred['logits']), feed_dict['y'].to(torch.float32))
            labels = feed_dict['y']
            correct= sum((torch.tensor(pred['pred']) == labels).float())/len(labels)
        elif self.task_type == 'regression':
            y = feed_dict['y']
            result = metric(pred['pred'], y)
            r2 = r2_score(y, pred['pred'])
            
        elif self.task_type == 'cox':
            result = metric(pred['logits'], feed_dict['E'], 'noties') 
            val_CI = calc_concordance_index(pred['logits'].detach().cpu().numpy(), 
                    feed_dict['E'].detach().cpu().numpy(), feed_dict['T'].detach().cpu().numpy())
            result = as_float(result)
        else:
            raise NotImplementedError()

        if meters is not None:
            meters.update({mode+'_loss':result})
            if self.task_type=='cox':
                meters.update({mode+'_CI':val_CI})
            if self.task_type == 'classification':
                meters.update({mode+'_accuracy':correct})
            if self.task_type == 'regression':
                meters.update({mode+'_r2':r2})

    def validate(self, data_loader, metric, meters=None, mode='valid'):
        if meters is None:
            meters = Meters()

        self._model.eval()
        for fd in data_loader:
            self.validate_step(fd, metric, meters=meters, mode=mode)

        return meters.avg



    def score(self, X, y, best=False, validate=False):
        if self.task_type == "cox":
            sort_idx = np.argsort(y['T'])[::-1]
            X=X[sort_idx]
            y={'E':np.array(y.iloc[sort_idx]['E']), 'T':np.array(y.iloc[sort_idx]['T'])}
        data_loader = self.get_dataloader(X, y, shuffle=None)
        score_path = {}
        if best: 
            meters = Meters()
            if hasattr(self, "best_model"):
                self._model = self.best_model
                if self.drop_input:
                    best_index = self._index_path[self.best_lam_ind]
                    data_loader = self.get_dataloader(X[:, best_index], y, shuffle=None)
            self.validate(data_loader, self.metric, meters, mode='test')
            if self.task_type == "cox":
                return meters.avg['test_CI']
            elif self.task_type == "regression":
                return meters.avg['test_r2']
            else:
                return meters.avg['test_accuracy']
        else:
            msize=get_model_size(self._model_path)
            for i, model in enumerate(self._model_path):
                if not validate:
                    score_path[f'size{i}'] = msize[i]
                if np.isnan(msize[i]):
                    score_path[f'score{i}'] = np.nan
                    continue
                meters = Meters()
                self._model = model
                if self.drop_input:
                    data_loader = self.get_dataloader(X[:, self._index_path[i]], y, shuffle=None)
                self.validate(data_loader, self.metric, meters, mode='test')
                score_path[f'score{i}']=-meters.avg['test_loss']    
                # if self.task_type == "cox":
                #     score_path[f'score{i}']=meters._canonize_values("avg")['test_CI']                         
            return score_path


    def set_params(self, **params):
        if "lam" in params:
            self.lam = [params["lam"]]
        if "learning_rate" in params:
            self.learning_rate = params["learning_rate"]
        if "inner_penalty" in params:
            self.inner_penalty = params["inner_penalty"]
        if "alpha" in params:
            self.alpha = params["alpha"]
        if "weight_decay" in params:
            self.weight_decay = params["weight_decay"]
        if "hidden_dims" in params:
            self.hidden_dims = params["hidden_dims"]
        return self._init_nn()   


    def fit_and_validate(self, X, y, param_grid, test_size=0.2, n_jobs=None, nmin_features=None, nmax_features=None, **kwargs):
        """
        Randomly split data into training and validation sets. Then fit the GCRNet model to the training data and validate on the validation set.

        Args:
            X (array-like): The input training data of shape (n_samples, n_features).
            y (array-like): The target training data of shape (n_samples,) for regression and classification or (n_samples, 2) for survival analysis.
            param_grid (dict): The grid of hyperparameters to search during model training.
            test_size (float): The proportion of the dataset to include in the validation split. The default value is 0.2, indicating a 20% validation split. 
            n_jobs (int or None): The argument controls the parallel execution of the model fitting process, allowing multiple jobs to run simultaneously. Setting n_jobs to an integer value greater than 1 enables parallel execution. 
                The default value is None, meaning that only one job will run during model fitting.
            nmin_features (int or None): The minimum number of features to be selected. If None, there is no lower limit for feature selection. The default value is None.
            nmax_features (int or None): The maximum number of features to be selected. If None, there is no upper limit for feature selection. The default value is None.

        Returns:
            self (object): The best fitted GCRNet model based on the evaluation metric.
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random.randint(0, 100))
        param_list = get_param_combination(param_grid)
        if n_jobs is not None: 
            def _set_param_and_fit(model, params):
                model.set_params(**params)
                model.fit(X_train, y_train, **kwargs)
                score = model.score(X=X_test, y=y_test, validate=True)
                score = np.array(list(score.values()))
                size= get_model_size(model._model_path)
                if nmax_features is not None and nmin_features is not None:
                    target = np.all([size>= nmin_features, size<= nmax_features], axis=0)
                    if np.any(target):
                        score[np.invert(target)] = np.nan
                cur_max_score = np.nanmax(score)
                model.best_lam_ind = nanargmax_safe(score, axis=0)
                model.best_model = model._model_path[model.best_lam_ind]
                model.best_params = params
                return cur_max_score, model
            results = Parallel(n_jobs=n_jobs)(
                delayed(_set_param_and_fit)(model=copy.deepcopy(self), params=params) for params in param_list
            )
            score_array = [res[0] for res in results]
            model_array = [res[1] for res in results]
            ind_param = np.nanargmax(score_array, axis=0)
            return model_array[ind_param]
        else: 
            max_score=-np.inf
            best_model = None
            for params in param_list:
                self.set_params(**params)
                self.fit(X_train, y_train, **kwargs)
                score = self.score(X=X_test, y=y_test, validate=True)
                score = np.array(list(score.values()))
                size= get_model_size(self._model_path)
                if nmax_features is not None and nmin_features is not None:
                    target = np.all([size>= nmin_features, size<= nmax_features], axis=0)
                    if np.any(target):
                        score[np.invert(target)] = np.nan
                cur_max_score = np.nanmax(score)
                self.best_lam_ind = np.nanargmax(score, axis=0)
                self.best_model = self._model_path[self.best_lam_ind]
                self.best_params = params
                if max_score < cur_max_score:
                    max_score = cur_max_score
                    best_model = copy.deepcopy(self)
            return best_model


    def plot_solution_path(self, best=False, var_name=None, legend=False):
            n_model =  len(self._model_path)
            n_features = self.input_dim
            weight_path = np.zeros(shape=(n_features, n_model))
            with torch.no_grad():
                for j, _model in enumerate(self._model_path):
                    weight = _model.mlp[0][0].weight
                    if self.drop_input:
                        for i, ind in enumerate(self._index_path[j]):
                            weight_path[ind,j] = torch.norm(weight[:,i], p=2)
                    else:
                        for i in range(n_features):
                            weight_path[i,j] = torch.norm(weight[:,i], p=2)
            if var_name is None or len(var_name) != n_features :
                var_name = [f'x{i}' for i in range(n_features)]
            for i in range(n_features):
                plt.plot(np.log(self.lam),weight_path[i,:], label=var_name[i])
            if best:
                plt.axvline(x=np.log(self.lam[self.best_lam_ind]), linestyle='dashed')
            if legend:
                plt.legend()
            plt.xlabel(r"$log(\lambda)$")
            plt.ylabel(r"$\|\|\mathbf{W}_{0j}\|\|_2$")
            plt.show()

