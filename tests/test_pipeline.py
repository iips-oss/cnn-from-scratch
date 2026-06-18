import numpy as np
from src.layers import Conv, MaxPool, Flatten, Dense
from src.activations import GELU, ReLU

# simple test for entire forward and backward chain
def test_full_pipeline():
    img = np.random.randn(3, 32, 32)
    
    # define layers
    c = Conv((3, 32, 32), 3, 8)
    g = GELU()
    p = MaxPool(2, 2)
    f = Flatten()
    d = Dense(1800, 10)
    r = ReLU()

    # forward
    o_c = c.forward(img)
    o_g = g.forward(o_c)
    o_p = p.forward(o_g)
    o_f = f.forward(o_p[np.newaxis, :, :, :])
    out = d.forward(o_f)
    res = r.forward(out)

    # basic output shape check
    assert res.shape == (1, 10)

    # backward
    grad = np.random.randn(1, 10)
    g_r = r.backward(grad)
    g_d = d.backward(g_r)
    g_f = f.backward(g_d)
    g_p = p.backward(g_f[0])
    g_g = g.backward(g_p)
    g_c = c.backward(g_g)

    # gradient input shape check
    assert g_c.shape == (3, 32, 32)
