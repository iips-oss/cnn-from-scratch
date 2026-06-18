import numpy as np
from src.activations import GELU, ReLU

# simple test checks for gelu/relu forward and backward passes

def test_gelu_forward_exact():
    g = GELU(approximate='none')
    x = np.array([-5.0, 0.0, 5.0])
    out = g.forward(x)

    # 0 should stay 0
    assert abs(out[1]) < 1e-4
    # large positive should match the input
    assert abs(out[2] - 5.0) < 1e-2

def test_gelu_forward_approximate():
    g = GELU(approximate='tanh')
    x = np.array([-5.0, 0.0, 5.0])
    out = g.forward(x)

    # 0 should stay 0
    assert abs(out[1]) < 1e-4
    # large positive should match the input
    assert abs(out[2] - 5.0) < 1e-2

def test_gelu_backward_exact():
    g = GELU(approximate='none')
    x = np.array([-1.0, 0.0, 1.0])
    g.forward(x)
    
    grad = g.backward(np.ones_like(x))
    assert len(grad) == 3

def test_gelu_backward_approximate():
    g = GELU(approximate='tanh')
    x = np.array([-1.0, 0.0, 1.0])
    g.forward(x)
    
    grad = g.backward(np.ones_like(x))
    assert len(grad) == 3

def test_relu_forward():
    r = ReLU()
    x = np.array([-2.0, 3.0])
    out = r.forward(x)
    assert out[0] == 0.0
    assert out[1] == 3.0

def test_relu_backward():
    r = ReLU()
    x = np.array([-2.0, 3.0])
    r.forward(x)
    grad = r.backward(np.array([1.0, 1.0]))
    assert grad[0] == 0.0
    assert grad[1] == 1.0
