# cat-breed-classifier
This project classify a photo of a cat into one of **12 breeds** (`Abyssinian`, `Bengal`, `Birman`, `Bombay`, `British Shorthair`, `Egyptian Mau`, `Maine Coon`, `Persian`, `Ragdoll`, `Russian Blue`, `Siamese`, `Sphynx`) with a
convolutional neural network (CNN) that is built and trained here from random weights.
It follows by a fixed seed, clean train / val / test split, a baseline to beat,
regularisation, loss curves, a confusion matrix, one-shot test evaluation and an honest failure analysis.

## Task & Project Overview
* Build and train a custom CNN architecture from scratch that perform multi-class image classification on cat breeds while maintaining converging loss curves and closing the validation gap.
* **Dataset**: Oxford-IIIT Pet, only cats photos.

## Setup
Python 3.8 or higher
```powershell
pip install -r requirements.txt
```

## Load data
Downloads the Oxford-IIIT Pet dataset from Hugging Face, filters the 12 cat breeds, creates train/validation splits, and handles image resizing and RGB decoding.
```powershell
python -m src.data
```

## Check the model
Defines the CatCNN neural network architecture (4 stacked convolutional blocks, batch normalization, global average pooling, and dropout).
```powershell
python -m src.model
```

## Baselines
Benchmark performance against simple baseline models and save result [here](outputs/baselines.json).
```powershell
python -m src.baselines
```

## Train the Model
Executes model training under different configuration settings.
```powershell
python -m src.train --out outputs/cnn # baseline model (augmentation on)
```
```powershell
python -m src.train --no_aug --out outputs/cnn_noaug # experiment 1: no augmentation
```
```powershell
python -m src.train --dropout 0.5 --out outputs/cnn_dropout05 # experiment 2: more dropout
```
Summarize all runs in to [validation table](outputs/results.md).
```powershell
python -m src.summarize # keep in (outputs/results.md)
```

##  Evaluate
One-shot test evaluation loads the saved model checkpoint (best.pt) and evaluates it against the held-out test set. 
You can choose one of the performing run based from the [validation table](outputs/results.md) and only run the one you choose **ONCE**.
- if you choose **cnn**, run only this **once**.
```powershell
python -m src.evaluate --run outputs/cnn
```
- if you choose **cnn_dropout05**, run only this **once**.
```powershell
python -m src.evaluate --run outputs/cnn_dropout05
```
- if you choose **cnn_noaug**, run only this **once**.
```powershell
python -m src.evaluate --run outputs/cnn_noaug
```

## Try by your own
Choose your own photo and named as **'cat.jpg'**, to preprocesses an external image (converts RGB, resizes to 128×128, applies mean/std normalization) and runs a forward pass through CatCNN to predicted class labels and top-3 Softmax probability distributions.
```powershell
python -m src.predict  --run outputs/cnn --image cat.jpg # try one example photo named as 'cat.jpg'
```

## References
- [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) (CC BY-SA 4.0)
- [Hugging Face mirror timm/oxford-iiit-pet](https://huggingface.co/datasets/timm/oxford-iiit-pet)
- [cat.jpg](https://www.pinterest.com/pin/642959284345302529/)
