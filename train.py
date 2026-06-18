import numpy as np
import time
from src.data_loader import fetch_mnist
from src.layers import Conv, MaxPool, Flatten, Dense
from src.activations import GELU, ReLU

# simple adam class
class Adam:
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, wd=1e-4):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = wd
        self.t = 0
        
        self.m = {}
        self.v = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights') or hasattr(layer, 'kernels'):
                p = layer.weights if hasattr(layer, 'weights') else layer.kernels
                self.m[i] = np.zeros_like(p)
                self.v[i] = np.zeros_like(p)
                self.m[(i, 'b')] = np.zeros_like(layer.biases)
                self.v[(i, 'b')] = np.zeros_like(layer.biases)

    def step(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                dw = layer.dweights + self.wd * layer.weights
                db = layer.dbiases + self.wd * layer.biases
                
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * dw
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.square(dw)
                m_hat = self.m[i] / (1 - self.beta1**self.t)
                v_hat = self.v[i] / (1 - self.beta2**self.t)
                layer.weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                
                self.m[(i, 'b')] = self.beta1 * self.m[(i, 'b')] + (1 - self.beta1) * db
                self.v[(i, 'b')] = self.beta2 * self.v[(i, 'b')] + (1 - self.beta2) * np.square(db)
                m_hat_b = self.m[(i, 'b')] / (1 - self.beta1**self.t)
                v_hat_b = self.v[(i, 'b')] / (1 - self.beta2**self.t)
                layer.biases -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
                
            elif hasattr(layer, 'kernels'):
                dk = layer.dkernels + self.wd * layer.kernels
                db = layer.dbiases + self.wd * layer.biases
                
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * dk
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.square(dk)
                m_hat = self.m[i] / (1 - self.beta1**self.t)
                v_hat = self.v[i] / (1 - self.beta2**self.t)
                layer.kernels -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                
                self.m[(i, 'b')] = self.beta1 * self.m[(i, 'b')] + (1 - self.beta1) * db
                self.v[(i, 'b')] = self.beta2 * self.v[(i, 'b')] + (1 - self.beta2) * np.square(db)
                m_hat_b = self.m[(i, 'b')] / (1 - self.beta1**self.t)
                v_hat_b = self.v[(i, 'b')] / (1 - self.beta2**self.t)
                layer.biases -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)


def main():
    # Training Hyperparameters
    TRAIN_SIZE = 20000
    EPOCHS = 5
    LEARNING_RATE = 0.001
    LR_DECAY = 0.7
    WEIGHT_DECAY = 1e-4
    NUM_KERNELS = 12
    KERNEL_SIZE = 5

    print("loading mnist dataset...")
    x_train, y_train, x_test, y_test = fetch_mnist("data")
    
    # shuffle
    indices = np.random.permutation(len(x_train))
    x_train, y_train = x_train[indices], y_train[indices]
    
    x_train_sub = x_train[:TRAIN_SIZE]
    y_train_sub = y_train[:TRAIN_SIZE]
    
    # Calculate flat size dynamically
    conv_out_dim = 28 - KERNEL_SIZE + 1
    pool_out_dim = conv_out_dim // 2
    flat_size = NUM_KERNELS * pool_out_dim * pool_out_dim

    # setup network
    conv = Conv((1, 28, 28), KERNEL_SIZE, NUM_KERNELS)
    gelu = GELU()
    pool = MaxPool(2, 2)
    flat = Flatten()
    dense = Dense(flat_size, 10)
    
    layers = [conv, gelu, pool, flat, dense]
    opt = Adam(layers, lr=LEARNING_RATE, wd=WEIGHT_DECAY)
    
    print("training starting now")
    for epoch in range(EPOCHS):
        t0 = time.time()
        loss_val = 0
        correct = 0
        
        for i in range(TRAIN_SIZE):
            x = x_train_sub[i][np.newaxis, :, :]
            y = y_train_sub[i]
            
            # forward passes
            out1 = conv.forward(x)
            out2 = gelu.forward(out1)
            out3 = pool.forward(out2)
            out4 = flat.forward(out3[np.newaxis, :, :, :])
            out5 = dense.forward(out4)
            
            # softmax and cross entropy loss
            logits = out5[0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            loss_val += -np.log(probs[y] + 1e-15)
            if np.argmax(probs) == y:
                correct += 1
                
            # backprop
            grad = probs.copy()
            grad[y] -= 1.0
            grad = grad[np.newaxis, :]
            
            g5 = dense.backward(grad)
            g4 = flat.backward(g5)
            g3 = pool.backward(g4[0])
            g2 = gelu.backward(g3)
            conv.backward(g2)
            
            opt.step()
            
            if (i + 1) % 5000 == 0:
                print(f"epoch {epoch+1} progress {i+1}/{TRAIN_SIZE} loss={loss_val/(i+1):.4f}")
                
        # test accuracy
        test_correct = 0
        for i in range(len(x_test)):
            x = x_test[i][np.newaxis, :, :]
            y = y_test[i]
            
            out = dense.forward(flat.forward(pool.forward(gelu.forward(conv.forward(x)))[np.newaxis, :, :, :]))
            if np.argmax(out[0]) == y:
                test_correct += 1
                
        test_acc = test_correct / len(x_test) * 100
        train_acc = correct / TRAIN_SIZE * 100
        print(f"epoch {epoch+1} finished in {time.time()-t0:.1f}s. Loss={loss_val/TRAIN_SIZE:.4f}, train acc={train_acc:.2f}%, test acc={test_acc:.2f}%")
        
        opt.lr *= LR_DECAY

    # save trained weights
    print("saving trained weights...")
    np.savez_compressed("model_weights.npz", 
                        conv_kernels=conv.kernels, 
                        conv_biases=conv.biases,
                        dense_weights=dense.weights, 
                        dense_biases=dense.biases)
    print("model saved to model_weights.npz")


if __name__ == "__main__":
    main()
