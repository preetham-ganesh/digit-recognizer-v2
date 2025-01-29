# Digit Recognizer

This repository contains code used to train the Deep Learning models in the Digit Recognizer project.

## Contents

- [Installation](https://github.com/preetham-ganesh/digit-recognizer#installation)
- [Dataset](https://github.com/preetham-ganesh/digit-recognizer#dataset)
- [Usage](https://github.com/preetham-ganesh/digit-recognizer#usage)
- [Model Details](https://github.com/preetham-ganesh/digit-recognizer#model-details)
- [Support](https://github.com/preetham-ganesh/digit-recognizer#support)

## Installation

### Download the repo

```bash
git clone https://github.com/preetham-ganesh/digit-recognizer.git
cd digit-recognizer
```

### Requirements Installation

Requires: [Pip](https://pypi.org/project/pip/)

```bash
pip install --no-cache-dir -r requirements.txt
```

## Usage

Use the following commands to run the code files in the repo:

Note: All code files should be executed in home directory.

### Dataset

- The data was downloaded from Kaggle - Digit Recognizer competition [[Link]](https://www.kaggle.com/c/digit-recognizer/data).
- Due to the size of the data, it is not available in the repository.
- After downloading the data, 'train.csv' file should be saved in the following data directory path 'data/raw_data/digit_recognizer/train.csv'.

### Model Training & Testing

```bash
python3 src/digit_recognizer/train.py --model_version 1.0.0
```

or

```bash
python3 src/digit_recognizer/train.py -mv 1.0.0
```

### Serialization

```bash
python3 src/digit_recognizer/serialize.py --model_version 1.0.0
```

or

```bash
python3 src/digit_recognizer/serialize.py -mv 1.0.0
```

## Releases

Details about the latest releases, including key features, bug fixes, and any other relevant information.

| Version | Release Date | Release Notes                                                           |
| ------- | ------------ | ----------------------------------------------------------------------- |
| v1.0.0  | 03-10-2024   | Release Training, Testing & Serialization code (Sub-classing approach). |
