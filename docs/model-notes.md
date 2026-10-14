# Model Notes

These notes are derived from the supplied project notebooks.

## Input

Emoji PNG images are loaded and resized to 64 x 64 pixels with three RGB channels.

## Labels

Emoji names are extracted from image filenames and encoded with scikit-learn LabelEncoder. The supplied `label_classes.npy` contains 1,813 classes.

## CNN

The notebooks define three convolution blocks with 32, 64, and 128 filters, each followed by max pooling. The feature map is flattened and passed through a 128-unit ReLU dense layer and a softmax classification layer.

One notebook trains with a train/test split for 25 epochs and batch size 32. Another experiment trains on the prepared full dataset for 30 epochs. These are preserved as experiments rather than collapsed into a single unsupported training history.

## Inference

The notebooks resize the input to 64 x 64, expand a batch dimension, normalize by 255, run the saved Keras model, and map the predicted class index through the saved labels.
