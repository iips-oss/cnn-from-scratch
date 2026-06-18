import sys
import numpy as np
from PIL import Image

from src.layers import Conv, MaxPool, Flatten, Dense
from src.activations import GELU


def main():
    if len(sys.argv) < 2:
        print("error: please provide image path. Usage: python predict.py <image_path>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    
    # load model weights
    try:
        data = np.load("model_weights.npz")
    except FileNotFoundError:
        print("error: model_weights.npz not found! run train.py first to train and save the model.")
        sys.exit(1)
        
    # define model
    conv = Conv((1, 28, 28), 5, 12)
    gelu = GELU()
    pool = MaxPool(2, 2)
    flat = Flatten()
    dense = Dense(1728, 10)
    
    # load weights to layers
    conv.kernels = data["conv_kernels"]
    conv.biases = data["conv_biases"]
    dense.weights = data["dense_weights"]
    dense.biases = data["dense_biases"]
    
    # load custom image
    try:
        img = Image.open(img_path).convert('L')
    except Exception as e:
        print(f"error loading image: {e}")
        sys.exit(1)
        
    # resize to 100x100 to estimate background and crop accurately
    img = img.resize((100, 100))
    arr = np.array(img)
    
    # Invert and remove background noise dynamically
    if arr[0, 0] > 128:
        # light background
        bg_noise = max(arr[0, 0], arr[-1, -1], arr[0, -1], arr[-1, 0])
        arr[arr > bg_noise - 10] = bg_noise
        arr = bg_noise - arr
    else:
        # dark background
        bg_noise = min(arr[0, 0], arr[-1, -1], arr[0, -1], arr[-1, 0])
        arr[arr < bg_noise + 10] = bg_noise
        arr = arr - bg_noise
        
    # Bounding box crop and center (same as MNIST dataset preprocessing)
    non_zero = np.argwhere(arr > 35)
    if len(non_zero) > 0:
        min_y, min_x = non_zero.min(axis=0)
        max_y, max_x = non_zero.max(axis=0)
        cropped = arr[min_y:max_y+1, min_x:max_x+1]
        
        # clean background inside the cropped box
        cropped[cropped < 45] = 0
        
        h, w = cropped.shape
        cropped_img = Image.fromarray(cropped)
        if h > w:
            new_h = 20
            new_w = int(20 * w / h)
        else:
            new_w = 20
            new_h = int(20 * h / w)
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        resized = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        canvas = Image.new('L', (28, 28), 0)
        offset_x = (28 - new_w) // 2
        offset_y = (28 - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))
        arr = np.array(canvas)
        
    # normalize to [-0.5, 0.5]
    x = (arr / 255.0) - 0.5
    x = x[np.newaxis, :, :] # add channel dimension -> (1, 28, 28)
    
    # forward pass
    out1 = conv.forward(x)
    out2 = gelu.forward(out1)
    out3 = pool.forward(out2)
    out4 = flat.forward(out3[np.newaxis, :, :, :])
    logits = dense.forward(out4)[0]
    
    # predict digit
    pred = np.argmax(logits)
    print(f"prediction: {pred}")


if __name__ == "__main__":
    main()
