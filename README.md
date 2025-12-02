# SFU CMPT 459: Data Mining | Plant Health Classification

# Overview
This project uses machine learning, data analysis, and a custom classifier to predict the stress level of the plant. Our trained models can give automated warnings when a plant is likely experiencing stress. This is valuable because gardeners often notice issues too late, which can lead to smaller harvests or even plant loss. The data can be collected with affordable equipment such as soil probes, light sensors, and temperature monitors. Also, we can analyze this data to discover patterns and relationships, such as how soil quality, water levels, and other features affect growth and plant stress. In the future, the trained models could run on IoT devices for real-time monitoring. This would result in a smart garden assistant that supports healthier plants, reduces waste, and promotes sustainable living.

Dataset: https://www.kaggle.com/datasets/ziya07/plant-health-data

# Installation
## Create and activate a virtual environment
```
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

## Install the project
```
pip install -e .
```

```
# Project Structure
```text
├── data/                         # Pre-split datasets
├── eda/                          # Exploratory data analysis notebooks
├── src/                          # Python source code
│   ├── alex/                     # Alex's models
│   ├── aki/                      # Aki's models 
│   ├── isaac/                    # Isaac's models
│   ├── extraction.py             # Shared feature-extraction utilities
│   └── preprocessing.py          # Shared preprocessing utilities
├── submission/                   # Final notebooks for grading
│   ├── classification/
│   ├── clustering/
│   ├── feature_selection/
│   └── outlier-detection/
├── README.md
├── requirements.txt              # Python dependencies
└── setup.py                      # Package setup (for training / installs)
```

# Final Submission Notebooks
We split up the final submission notebooks into 3 folders, one for each main task we performed. 
- Inside `submission/classification`, you will find notebooks for our `KNN`, `SVM`, and `Random Forest` classifiers.
- Inside `submissions/clustering`, you will find notebooks for our `DBSCAN`, `EM`, and `Hierarchical` clustering algorithms.
- Inside `submissions/outlier-detection`, you will find notebooks for our `Kernel Density`, `Local Outlier Factor`, and `Isolation Forest` outlier detection algorithms.

## K-Nearest Neighbour (K-NN) Inference
1.) Make sure to check out the correct branch.
```
git checkout knn-inference
```

2.) Train the model by running the notebook in `./knn/train.ipynb`.

3.) Start a simple web app for inference:
```
source venv/bin/activate
cd ./knn && ./inference.sh
```

4.) Open your browser and visit: http://localhost:5000
On the web app, you can select 10 different plants that were originally from the dataset.
We built a cubic spline interpolator for each plant (inspired by SFU MACM 316 interpolation) to generate samples for demonstration purposes.
Then, you can hover over the plot, and an inference will run given the 8 selected parameters and would classify whether or not the selected plant with certain attributes is considered `Healthy`, `Moderately Stress`, or `High Stress`.
