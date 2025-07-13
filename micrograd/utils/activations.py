__author__ = "Anant Sakhare"
__version__ = "v0.1"

import numpy as np
import torch 

def step(x : float  | int):
    return 1 if x>= 0 else 0

def sigmoid(x :  np.ndarray |  torch.Tensor):
    return 1 / (1 + np.exp(-x))

def tanh(x : np.ndarray |  torch.Tensor):
    return 2 / (1 + np.exp(-2* x)) -1 

def relu(x : np.ndarray |  torch.Tensor):
    return np.where(x>=0, x, 0)


def leaky_relu(x : np.ndarray |  torch.Tensor, alpha=0.001):
    return np.where(x>=0, x, x * alpha)

def softmax(x : np.ndarray |  torch.Tensor):
    exps = np.exp(x  -  np.max(x))   # avoided below line part for it.
    return exps / np.sum(exps)  # np.exp(x) /  sum(np.exp(x)) can lead to overflow for large values


if __name__ == "__main__":

    val = 0.5
    arr = np.array([0.2,0.3,0.4])
