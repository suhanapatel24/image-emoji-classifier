# Image-Based Emoji Classifier

A convolutional neural network project for classifying emoji images into **1,813 label classes** using TensorFlow/Keras.

This repository is organized from the original project notebooks and trained artifacts. The original experiments resize emoji PNGs to 64x64 RGB images, encode emoji names as class labels, and train a CNN classifier.

## Model architecture

The training notebooks define:

- Conv2D(32) + MaxPooling
- Conv2D(64) + MaxPooling
- Conv2D(128) + MaxPooling
- Flatten
- Dense(128)
- Softmax output over the encoded emoji classes

## Inference preprocessing

1. Convert the uploaded image to RGB.
2. Resize to 64x64.
3. Convert to a NumPy array.
4. Normalize pixel values to [0, 1].
5. Run the trained Keras model.
6. Map output indices through `label_classes.npy`.

## Run the Streamlit app

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Place the supplied `emoji_classifier_model.h5` and `label_classes.npy` in the repository root.

## Project structure

```text
app.py                       Streamlit inference UI
src/inference.py             reusable preprocessing/prediction logic
notebooks/                   original exploratory/training notebooks
emoji_classifier_model.h5    trained model (uploaded separately)
label_classes.npy            encoded label classes (uploaded separately)
requirements.txt
```

## Notes on reproducibility

The original notebooks reference a local emoji image dataset path that was not included with the uploaded project files. For that reason, this repository preserves the original notebooks and trained artifacts but does not claim that training can be reproduced from scratch without the source image dataset.

No accuracy figure is stated here because the supplied files do not provide a single verified final evaluation metric suitable for a resume claim.
