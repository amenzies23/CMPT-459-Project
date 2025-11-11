# CMPT-459-Project

# Layout
```
├── data
├── eda
├── README.md
├── requirements.txt
├── setup.py
├── src
```

# Installation
We designed our Project as a python package. All our training 
and pre-processing code is in the ```src/``` directory.
To install the project and get running. Run 

```
pip install -e .
```

# Basic Usage
In any python file you can now use
```python
from preprocessing import preprocess
import pandas as pd
df = pd.read_csv("data/plant_healt_data.csv")
model_ready = preprocess(df)
print(model_ready.head())
```

# Dataset
* link to project proposal

# Preprocessing
* What we do
* Why

# Feature Extraction
* Wavelet Transform

# Models