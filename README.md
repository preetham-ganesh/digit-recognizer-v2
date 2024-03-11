# Digit Recognizer v2 (Training)

This repository contains code used to train the Deep Learning models in the Digit Recognizer project.

## Contents

- [Installation](https://github.com/preetham-ganesh/digit-recognizer-v2-training#installation)
- [Usage](https://github.com/preetham-ganesh/digit-recognizer-v2-training#usage)
- [Releases](https://github.com/preetham-ganesh/digit-recognizer-v2-training#releases)

## Installation

### Download the repo

```bash
git clone https://github.com/preetham-ganesh/digit-recognizer-v2-training
cd digit-recognizer-v2-training
```

### Requirements Installation

Requires: [Pip](https://pypi.org/project/pip/)

```bash
pip install --no-cache-dir -r requirements.txt
```

## Usage

Use the following commands to run the code files in the repo:

Note: All code files should be executed in home directory.

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
