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

## Dataset

- The data was downloaded from Kaggle - Digit Recognizer competition [[Link]](https://www.kaggle.com/c/digit-recognizer/data).
- After downloading the data, 'train.csv' file should be saved in the following data directory path 'data/raw_data/train.csv'.

## Usage

Use the following commands to run the code files in the repo:

Note: All code files should be executed in home directory.

### Model Training & Testing

```bash
python3 src/run.py --experiment_name digit_recognizer --model_version 1.0.0
```

or

```bash
python3 src/run.py -en digit_recognizer -mv 1.0.0
```

## Model Details

| Model Name       | Model Version | Description                                    | Performance      | Model URL                                                                     |
| ---------------- | ------------- | ---------------------------------------------- | ---------------- | ----------------------------------------------------------------------------- |
| Digit Recognizer | v1.0.0        | A CNN model that recognizes digit in an image. | Accuracy: 98.54% | [Hugging Face](https://huggingface.co/preethamganesh/digit-recognizer-v1.0.0) |

## Support

For any queries regarding the repository please contact 'preetham.ganesh2021@gmail.com'.
