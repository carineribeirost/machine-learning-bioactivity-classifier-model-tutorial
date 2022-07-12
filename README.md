# Machine Learning Bioactivity Tutorial - Classifier Model

Notebook repository focused in the development of an AI Drug Discovery
machine learning model from scratch. 

## [Data Collection and Pre-Processing](https://github.com/carineribeirost/machine-learning-bioactivity-predictor-regression-model/blob/main/1_Data_Pre_Preparation.ipynb)
Collect and pre-process biological activity data from
ChEMBL database.

## [Morgan Descriptors Calculation](https://github.com/carineribeirost/machine-learning-bioactivity-predictor-regression-model/blob/main/2_Descriptors_Calculation_Morgan_Fingerprint.ipynb)
Compute the molecular descriptors using RDKit package and 
prepare the dataset (X and Y dataframes) that will be used for Model Building.

## [Descriptors Selection](https://github.com/carineribeirost/machine-learning-bioactivity-predictor-regression-model/blob/main/3_High_Colinearity_Low_Variance.ipynb)
Perform the removal of high colinearity and low variance features from the 
previously calculated descriptors.

## [Model Building and Evaluation](https://github.com/carineribeirost/machine-learning-bioactivity-predictor-regression-model/blob/main/4_classifier_ensemble_model.ipynb)
Use the computed molecular descriptors (X variables) 
to build a classifier model for predicting the bioactivity class (Y variable).
The code also includes: nested and k-fold cross-validation.

## [Exploratory Data Analysis - SOON](https://github.com/carineribeirost) 
Perform exploratory data analysis by making simple box 
and scatter plots to discern differences of the 
active and inactive sets of compounds and Visualize the 
chemical space using PCA and t-SNE 


## Libraries used

* [Pandas](https://pandas.pydata.org/) - python package for easy and intuitive data manipulation and analysis

* [NumPy](https://numpy.org/) -  the fundamental package for array computing with Python

* [RDKit](https://www.rdkit.org/) - Open source toolkit for cheminformatics

* [Scikit-learn](https://scikit-learn.org/stable/) - Machine Learning in Python.

* [Matplotlib](https://matplotlib.org/) - a comprehensive library for creating static, animated, and interactive visualizations in Python.

* [Seaborn](https://seaborn.pydata.org/) - a Python data visualization library based on matplotlib.  

* [Plotly](https://plotly.com/) - provides graphing, analytics, and statistics tools, as well as scientific graphing libraries for Python, R and other languages.


Libraries were used in a [Conda3](https://docs.conda.io/en/latest/) environment using python 3.10.4

## Installation

Conda3: [Installation](https://docs.anaconda.com/anaconda/install/index.html)

pandas:
```
conda install -c anaconda pandas
```
numpy
```
conda install -c anaconda numpy
```
RDKit
```
conda install -c rdkit rdkit
```
scikit-learn
```
conda install -c anaconda scikit-learn
```
matplotlib
```
conda install -c conda-forge matplotlib
```
seaborn
```
conda install -c anaconda seaborn
```
plotly
```
conda install -c anaconda plotly
```
streamlit
```
conda install -c anaconda streamlit
```
## How to run
* Download the notebooks on the desired directory
```
conda jupyter notebook 
```
Select the chosen directory and file

## Pipeline used

* [ChEMBL Structure Pipeline](https://github.com/ChEMBL_Structure_Pipeline) - a protocol to standardize and salt strip molecules

[Installation](https://github.com/ChEMBL_Structure_Pipeline/blob/master/README.md)

## Observations

These notebooks have been elaborated using 
as references the following articles and codes:

* [Trust, But Verify: On the Importance of Chemical Structure Curation in Cheminformatics and QSAR Modeling Research](https://pubs.acs.org/doi/10.1021/ci100176x)

* [Probing the origins of human acetylcholinesterase inhibition via QSAR modeling and molecular docking](https://pubmed.ncbi.nlm.nih.gov/27602288/)

* [Python codes by Data Professor](https://github.com/dataprofessor/code/tree/master/python)

* [Hyperparameter Tuning the Random Forest in Python](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)

* [Repeated k-Fold Cross-Validation for Model Evaluation in Python](https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/)

* [Nested Cross-Validation for Machine Learning with Python](https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/)

## Authorship
* Author: **Carine Ribeiro** ([carineribeirost](https://github.com/carineribeirost))

