import pytest
import numpy as np

from micrograd.utils.activations import *

test_cases = [
    (step, 0, 1),
    (step, 0.5, 1),
    (step, -0.5, 0),
    (step, 1, 1),

    (sigmoid, 0, 0.5),
    (sigmoid, 0.5, 0.622459),
    (sigmoid, -0.5, 0.377540),
    (sigmoid, 1, 0.731059),

    (tanh, 0, 0.0),
    (tanh, 0.5, 0.462117),
    (tanh, -0.5, -0.462117),
    (tanh, 1, 0.761594),

    (relu, 0, 0),
    (relu, 0.5, 0.5),
    (relu, -0.5, 0),
    (relu, 1, 1),

    (leaky_relu, 0, 0.0),
    (leaky_relu, 0.5, 0.5),
    (leaky_relu, -0.5, -0.0005),
    (leaky_relu, 1, 1.0),
]

@pytest.mark.parametrize("func, x, expected", test_cases)
def test_activation(func, x, expected):
    output =  func(x)
    assert np.isclose(output, expected ,atol=1e-5)

