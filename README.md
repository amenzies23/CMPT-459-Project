# CMPT-459-Project

<<<<<<< HEAD
# Installation

## Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows

## Install Dependencies
For KNN Classification
```
pip instal -r aki_requirements.txt
```

=======
# Layout
>>>>>>> main
```
├── data
├── eda
├── README.md
├── requirements.txt
├── setup.py
├── src
```

<<<<<<< HEAD
### K-Nearest Neighbour (KNN) Classifier
Make sure you are in the correct directory:
```bash
cd ./knn
```

With Hyperparameter Tuning (default)
```bash
python main.py
```

Without Hyperparameter Tuning
```bash
python main.py --no-hyperparameter-tuning
```

During training, the script:
- Generates PCA and clustering visualizations
- Selects optimal features
- Saves the trained k-NN model, scaler, and label encoder under `./model/`

### Inference via Web App
After training completes:
```bash
python app/app.py
```
Then open your browser and visit: http://localhost:5000
Use the interactive sliders to change feature values and see predicted class labels.

### Clean Cache and Generated Files (Cleanup)
Run this to remove:
- Python cache files (__pycache__, .pkl, .png, etc.)
- Generated plots and model files
```bash
./clean.sh
```

### Train + Run Inference (one step)
To retrain the model with hyperparameter tuning and start the Flask app immediately:
```bash
./run.sh
```

### KNN Structure
.
├── app/
│   ├── app.py                # Flask web app for inference
│   └── templates/            # HTML templates
├── core/
│   ├── preprocessing.py
│   ├── outlier_detection.py
│   ├── feature_selection.py
│   ├── clustering.py
│   └── classification.py
├── model/                    # Saved model, scaler, encoder
├── main.py                   # Main training + visualization script
|── utils.py
├── clean.sh                  # Cleanup script
└── run.sh                    # Train + inference launcher

TODO: Revise as we work on the project.
TODO: Dataset, Preprocessing, Feature Extraction, Classifcation, etc..
...
=======
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
>>>>>>> main
