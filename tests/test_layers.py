import numpy as np
from src.layers import MaxPool, Dense, Conv, Flatten


def test_flatten():
    sample_input = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    flatten = Flatten()
    result = flatten.forward(sample_input)

    expected_output = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    np.testing.assert_array_equal(result, expected_output)

    grad_output = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    grad_input = flatten.backward(grad_output)

    np.testing.assert_allclose(grad_input, sample_input * 0.1, rtol=1e-7)



def test_maxpool():
    print("Max Poll Test:\n")
    sample_input = np.array([[[1, 3, 2, 4], [5, 6, 1, 2], [7, 8, 9, 3], [4, 5, 2, 1]]])
    pool = MaxPool(pool_size=2, stride=2)
    result = pool.forward(sample_input)

    expected_output = np.array([[[6, 4], [8, 9]]])
    np.testing.assert_array_equal(result, expected_output)


def test_dense():
    dense = Dense(input_size=3, output_size=2)
    dense.weights = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    dense.biases = np.array([[0.1, 0.2]])

    sample_input = np.array([[1.0, 1.0, 1.0]])
    out = dense.forward(sample_input)

    expected_out = np.array([[1.0, 1.4]])
    np.testing.assert_allclose(out, expected_out, rtol=1e-6)

    output_gradient = np.array([[1.0, 2.0]])
    input_gradient = dense.backward(output_gradient, learning_rate=0.01)

    expected_input_grad = np.array([[0.5, 1.1, 1.7]])
    np.testing.assert_allclose(input_gradient, expected_input_grad, rtol=1e-6)


def test_conv():
    image = np.random.randn(3, 32, 32)
    conv = Conv(
        input_shape=(3, 32, 32),
        kernel_size=3,
        num_kernels=8,
    )

    out = conv.forward(image)
    assert out.shape == (8, 30, 30), f"Expected shape (8, 30, 30), got {out.shape}"

    grad = np.random.randn(*out.shape)
    back = conv.backward(grad, learning_rate=0.01)
    assert back.shape == image.shape, (
        f"Expected backward shape {image.shape}, got {back.shape}"
    )


def test_conv_pool_pipeline():
    image = np.random.randn(3, 32, 32)
    conv = Conv(
        input_shape=(3, 32, 32),
        kernel_size=3,
        num_kernels=8,
    )
    pool = MaxPool(
        pool_size=2,
        stride=2,
    )

    conv_out = conv.forward(image)
    pool_out = pool.forward(conv_out)

    conv_out = conv.forward(image)
    pool_out = pool.forward(conv_out)

    assert conv_out.shape == (8, 30, 30), (
        f"Conv output shape mismatch: {conv_out.shape}"
    )
    assert pool_out.shape == (8, 15, 15), (
        f"Pool output shape mismatch: {pool_out.shape}"
    )
