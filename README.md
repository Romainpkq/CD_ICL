## Overview
This code is for the paper _Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding_. Our code is based on the <a href="https://github.com/Shark-NLP/OpenICL/tree/main">OpenICL repository</a>.

## Installation
Note: OpenICL requires Python 3.8+


**Installation for local development:**
```
git clone https://github.com/Romainpkq/CD_ICL.git

cd CD_ICL
pip install -e .
```

## Examples
Following example shows you how to perform ICL on sentiment classification dataset. 
```
# predict
bash run/run_origin.sh
```

```python
# calculate the accuracy
python run/predict.py
```
