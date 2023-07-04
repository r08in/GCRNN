# GCRNet: Sparse-Input Neural Networks with Group Concave Regularization

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
