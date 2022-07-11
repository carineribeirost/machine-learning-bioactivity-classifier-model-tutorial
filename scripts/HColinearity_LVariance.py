import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import matplotlib.pyplot as plt

def high_col_removal(dataframe, threshold):
    """
    Remove Columns with Colinearity Higher than a given threshold using a correlation matrix method
    """

    # Build a pearson correlation matrix
    cor_matrix = dataframe.corr(method ='pearson').abs()
    
    # Convert the matrix in a upper triangular Matrix given the colinearity simmetry
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
    
    # Write a list of columns with high colinearity to be dropped from original dataset
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    # Drop Selected columns and store dataframe
    dataframe = dataframe.drop(to_drop, axis=1)
    
    return dataframe

#This threshold is choosen considering the data splitting in 80/20
t = 0.8 * (1 - 0.8)
    

def remove_low_variance(input_data, threshold = t):
    """
    Remove columns from dataframe with Variance inferior to a given threshold
    """
    
    #Create a transform to be applied to dataframe (input_data)
    selection = VarianceThreshold(threshold)
    
    #Apply Transform to Dataframe
    selection.fit(input_data)
    
    
    return input_data[input_data.columns[selection.get_support(indices=True)]]
    
    
def correlation_matrix(dataframe):
    plt.matshow(dataframe.corr(method = 'pearson').abs())
    plt.show()