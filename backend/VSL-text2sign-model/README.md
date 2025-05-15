# Sign Speak Model

An artificial intelligence model for sign language translation.

## Overview

Sign Speak is a deep learning model designed to convert text to skeleton. The project uses an encoder-decoder architecture with transformer layers.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

- `model.py`: Main model definition
- `encoder.py` & `decoder.py`: Encoder-decoder components
- `transformer_layers.py`: Implementation of transformer layers
- `data.py`: Input data processing
- `training.py`: Model training
- `prediction.py`: Generate predictions from trained model
- `vocabulary.py`: Vocabulary management
- `dtw.py`: Dynamic Time Warping algorithm implementation

## Usage

### Training the Model

To run, start __main__.py with arguments "train" to train "test" to test and ".\Configs\Base.yaml":

`python __main__.py train ./Configs/Base.yaml`

## Dataset

Examples can be found in /Data/tmp. Data path must be specified in config file.

The dataset can be accessed at: [VSL-Text Sign Dataset](https://www.kaggle.com/datasets/pluslienminhchatgpt/vsl-text-sign-dataset)

### Input Data Format

The dataset includes matching pairs of:
- `.npy` files containing sign language features
- `.txt` files containing corresponding text translations