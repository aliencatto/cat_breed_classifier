# cat-breed-classifier
This project classify a photo of a cat into one of **12 breeds** (`Abyssinian`, `Bengal`, `Birman`, `Bombay`, `British Shorthair`, `Egyptian Mau`, `Maine Coon`, `Persian`, `Ragdoll`, `Russian Blue`, `Siamese`, `Sphynx`) with a
convolutional neural network (CNN) that is built and trained here from random weights.
It follows by a fixed seed, clean train / val / test split, a baseline to beat,
regularisation, loss curves, a confusion matrix, one-shot test evaluation and an honest failure analysis.

## Rubric checklist
 - [x] Real dataset, real model. A public dataset (cited) and a model you defined and trained — not a pre-trained model called as-is.
 - [ ] Honest train / val / test discipline. A clean 3-way split; the test set was touched exactly once, at the end. No hyperparameter was chosen by looking at test.
 - [ ] A baseline, beaten. You state a trivial baseline (majority-class / mean / linear) and your model beats it on the held-out test set.
 - [ ] At least one form of regularization (dropout, weight decay, augmentation, early stopping) — and you can say why.
 - [ ] Loss curves + a confusion matrix / metric. Train-vs-val curves you can read, plus a confusion matrix (or per-class metric / generated samples) — not just one accuracy number.
 - [x] A fixed seed so your numbers reproduce.
 - [ ] A short written report (the template below), including one honest failure case.

## Task & Project Overview
* Build and train a custom CNN architecture from scratch that perform multi-class image classification on cat breeds while maintaining converging loss curves and closing the validation gap.
* **Dataset**: Oxford-IIIT Pet, only cats photos.

## Setup
Python 3.8 or higher
```powershell
pip install -r requirements.txt
```

## Load data
Downloads the Oxford-IIIT Pet dataset from Hugging Face, filters the 12 cat breeds, creates stratified train/validation splits, and handles image resizing and RGB decoding.
```powershell
python -m src.data
```

## Check the model
Defines the CatCNN neural network architecture (4 stacked convolutional blocks, batch normalization, global average pooling, and dropout).
```powershell
python -m src.models
```

## References
- Oxford-IIIT Pet: https://www.robots.ox.ac.uk/~vgg/data/pets/ (CC BY-SA 4.0)  
- Hugging Face mirror timm/oxford-iiit-pet: (https://huggingface.co/datasets/timm/oxford-iiit-pet)