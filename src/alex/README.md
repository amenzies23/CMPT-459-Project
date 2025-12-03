# Alex's Directory
## Installation
After completing the initial installation of our project in the root `README`, my files should all be able to run with no extra dependencies.
## Explanation of files
### clustering.py
`marimo edit clustering.py`

This file goes through the full process of implementing EM Clustering using Gaussian Mixture. This was exported to a jupyter notebook pre-rendered in the final submission folder.
### isolation_forest.py
`marimo edit isolation_forest.py`

This file goes through the full process of implementing Isolation Forest outlier detection. This was exported to a jupyter notebook pre-rendered in the final submission folder.
### random_forest.py
`python src/alex/random_forest.py`

This file includes a full Random Forest Classifier analysis of our dataset. This walks through all scenarios from using a base classifier with no parameter tuning or feature selection, to clasifying with parameter tuning and feature selection. The goal of this file is to compare the performance across all variations of using the Random Forest classifier.

### random_forest_analysis.ipynb
This notebook shows a pre-rendered visual walk through of the above file. It visualizes plots for all of these runs, and compares the classifier variations in a cleaner way than looking at the console output. I took the best version of these classifiers to be used in the submission `random_forest.ipynb` file. Although the one difference between these files is the feature selection. I used Mutual Information here, and decided to implement Recursive Feature Elimination in the submission notebook. We saw equivalent results between both methods.
