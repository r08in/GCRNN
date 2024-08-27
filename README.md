# GCRNet: Sparse-Input Neural Networks with Group Concave Regularization

[Project Page](https://github.com/r08in/GCRNN)|[Paper](https://arxiv.org/abs/2307.00344)

GCRNet is a Python library that implements a novel framework for sparse-input neural networks using group concave regularization. Leveraging the power of concave penalties, specifically MCP and SCAD, GCRNet provides a comprehensive approach for simultaneous feature selection and non-linear function estimation. The proposed framework considers all outgoing connections from a single input neuron as a group and applies an appropriate concave penalty to the $l_2$ norm of weights within each group. By selectively shrinking the weights of certain groups to exact zeros, GCRNet constructs a neural network that utilizes only a small subset of variables, enhancing both the accuracy and interpretability of the feature selection process. GCRNet offers versatile functionality, supporting regression or classification tasks with continuous, binary, or time-to-event outcomes.

### Installation

#### Installation with pip

To install with `pip`, run the following command:
```
pip install --user gcrnet
```

#### Installation from GitHub

You can also clone the repository and install manually:
```
git clone 
cd gcrnet/python
python setup.py install --user
```

### Usage

Once you install the library, you can import `GCRNet` to create a model instance:
```
from gcrnet import GCRNet
lam=np.exp(np.linspace(np.log(0.01),np.log(0.5), 50))
gmcp_net = GCRNet(task_type='regression',device=torch.device("cpu"), input_dim=train_data['X'].shape[1], output_dim=1, hidden_dims=[10,5], activation="relu",
    optimizer="Adam",learning_rate=0.001, batch_size=train_data['X'].shape[0], 
    alpha=0.01, lam=lam, penalty="MCP",drop_input=True)

# fit model and tune parameters
param_grid={"alpha":[0.03]}
gmcp_net= gmcp_net.fit_and_validate(X=train_data['X'], y=train_data['y'], 
                              param_grid=param_grid, init_num_epochs=2000, num_epochs=200, verbose=True, print_interval=200)
```
### Regression/Classification/Survival Examples
Please see the examples in our Colab notebooks:

- [Regression example](https://colab.research.google.com/github/r08in/GCRNN/blob/main/python/examples/Regression-example.ipynb)
- [Classification example](https://colab.research.google.com/github/r08in/GCRNN/blob/main/python/examples/Classification-example.ipynb)
- [Cox example](https://colab.research.google.com/github/r08in/GCRNN/blob/main/python/examples/Cox-example.ipynb)

### Multivariate Survival Analysis Example
We extend our framework to simultaneously select common variables and estimate models for multivariate failure time data. Our approach employs the penalized pseudo-partial likelihood method within a non-linear marginal hazard model. Specifically, each marginal hazard function is approximated by a feed-forward neural network with an identical input layer. We treat all outgoing connections from a single input neuron across feed-forward neural networks as a collective group and apply a concave penalty to the  $l_2$  norm of the weights within each group. By shrinking the weights of specific groups to exact zeros, our method yields a collection of neural networks that utilize only a concise subset of common variables. This group-level penalization facilitates the inclusion or exclusion of entire groups of parameters, enabling the identification of common variables relevant to multivariate failure time data.
See the example in our Colab notebooks:
- [Bivariate Cox example](https://colab.research.google.com/github/r08in/GCRNN/blob/main/python/examples/Bivariate-Cox-example.ipynb)

### Acknowledgements and References

Some of our codebase and its structure are inspired by https://github.com/runopti/stg. 

If you find our library useful in your research, please consider citing us:

```
@misc{luo2023sparseinput,
      title={Sparse-Input Neural Network using Group Concave Regularization}, 
      author={Bin Luo and Susan Halabi},
      year={2023},
      eprint={2307.00344},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
