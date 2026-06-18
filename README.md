# cnn-from-scratch

## Load dataset
To load the dataset go inside the `src/` folder and run `data_loader.py` file
The data will get downloaded and store inside the `data/` folder, if not present will get created automatically.

## Requirements
to install the requirements first install the `uv` installer 
and run `uv sync` this will automatically create env as `.venv` and install the depedecies. :)

## Testing Layers
For a quick riview and testing, `tests/` have test functions just run the `uv run pytest` command to test all the contstucted layers and activations. :)

## Training the CNN
To start training, simply run:
`uv run train.py`
You can edit hyperparameters (epochs, learning rate, kernel size) at the top of the `main()` function inside `train.py`. The weights will automatically save to `model_weights.npz` when training completes!

## Predict Custom Digits
You can run predictions on your own handwritten digit images:
`uv run predict.py <path_to_image>`
The script will resize it to 28x28, convert to grayscale, and automatically invert the background if it is white (like pen on paper) so the model reads it correctly.
Also you can try ```img/``` folder to test some real world examples .:)

enjoy!!!!!


