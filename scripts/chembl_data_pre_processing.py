#Imports

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from chembl_webresource_client.new_client import new_client

from chembl_structure_pipeline import standardizer

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem.Draw import IPythonConsole

rdBase.rdkitVersion

from rdkit.Chem import MolStandardize

from IPython.display import display


#Functions
def print_columns(dataframe):
    for i in dataframe.columns:
        print(i)

def data_selection(target_select, standard_value):
    """
    Given an target name or search condition and a type of bioactivity (IC50, KI50) this function will create a dataframe
    with the selected target parameters
    """
    
    #access chembl database to search for desirable targets
    target = new_client.target
    target_query = target.search(target_select)
    targets = pd.DataFrame.from_dict(target_query)
    
    #show a dataframe with the possible targets for your search
    display(targets.filter(['target_chembl_id', 'pref_name']))
    
    #the index of the desirable target will be selected below
    index = int(input('please, select index after looking at the dataframe\n'))
    selected_target = targets.target_chembl_id[index] 
    
    activity = new_client.activity
    
    
    res = activity.filter(target_chembl_id=selected_target).filter(standard_type=standard_value)
    
    #build dataframe with selected target
    df = pd.DataFrame.from_dict(res)
    
    return df

def remove_mutation(dataframe):
    """
    This function will search for mutations on the dataframe and remove then
    """
    #verify if dataframe has a columns of mutation
    if 'assay_variant_mutation' in dataframe.columns:
        mutation_list = list() #list to store different mutations
        
        for i in dataframe.assay_variant_mutation.unique():
            mutation_list.append(i)
        mutation_list = mutation_list[1:] #the first value of this list is always none
        
        #drop the mutations from original dataframe
        for i in mutation_list:
            dataframe = dataframe.drop(dataframe[dataframe['assay_variant_mutation'] == i].index)
        
        return dataframe
    
def remove_assay_type(dataframe):
    """
    This function will look for different values of assay_type and ask if you want to remvoe then
    """
    display(dataframe.assay_type.value_counts())
    assay = ''
    while assay not in ['n', 'no', 'stop']:
        assay = input("Which assay_type do you wanna remove? type 'no' if none will be removed")
        dataframe = dataframe.drop(dataframe[dataframe['assay_type'] == assay].index)
    return dataframe


def remove_null_num_values(dataframe):
    """
    This function rmoves all none and non numerical rows and columns in the dataframe
    """
    dataframe = dataframe.drop(dataframe[dataframe['standard_relation'] == '>'].index)
    dataframe = dataframe.drop(dataframe[dataframe['standard_relation'] == '<'].index)

    dataframe = dataframe.dropna(subset = ['standard_value'])
    dataframe = dataframe.dropna(subset = ['standard_units'])
    dataframe = dataframe.dropna(subset = ['canonical_smiles'])
    dataframe = dataframe.dropna(axis=1, how='all')
    dataframe['standard_value'] = dataframe['standard_value'].astype(float)
    
    return dataframe
    
def dupe_removal(dataframe, interest_column = 'canonical_smiles', bioactivity_column = 'standard_value', threshold = 2):
    """
    Remove duplicated smiles from dataframe if standard deviation of their standard value is greater than a threshold
    """
    df_dupe = dataframe[dataframe[interest_column].duplicated(keep = False)]
    
    #Make a list with the SMILES representing each dupe
    smile_dupes = list()
    for i in df_dupe[interest_column]:
        if i not in smile_dupes:
            smile_dupes.append(i)
    
    #Make a list with Standard deviation for each set of duplicates
    deviation = list()
    for i in smile_dupes:
        df_dev = df_dupe[df_dupe[interest_column] == i]
        deviation.append([df_dev[bioactivity_column].std(), i]) #(bioactivity, SMILE)
        
    # List with all the duplicates with bioactivity deviation lesser than two, to be removed from the dataframe
    to_be_removed = list()
    for i, j in deviation:
        if i <= threshold:
            to_be_removed.append(j)
            
    #Removal of duplicated canonical smiles with small deviation from original dataset
    for i in to_be_removed:
        dataframe = dataframe.drop(dataframe.loc[dataframe[interest_column] == i].index)
    return dataframe


def smiles_standardizer(dataframe):
    """
    COnvert the canonical smiles of a given dataframe to their standardized version, using 
    chembl_structure_pipeline 
    """
    
    std_smiles_list = list()
    for i, smi in enumerate(dataframe.canonical_smiles):
        try:
            #converts to mol_block
            mol = Chem.MolFromSmiles(smi)
            mol_block = Chem.MolToMolBlock(mol)
            
            #standardize mol_block
            std_mol = standardizer.standardize_molblock(mol_block)
            mol = Chem.MolFromMolBlock(std_mol)
            
            #convert to smile
            smile = Chem.MolToSmiles(mol)
            std_smiles_list.append(smile)
        except:
            print(i, smi)
            
    # swap canonical smiles for std canonical smiles
    dataframe['canonical_smiles'] = std_smiles_list
    return dataframe
            

def normalize_activity(dataframe):
    """
    Normalize the dataframe standard value using a log10 factor
    """
    activity = list()
    
    #Exceedingly hogh values for bioactivity will skew the model statistics
    for i in dataframe.standard_value:
        if i > 100000000:
            i = 100000000
            
        #Converts units and normalize bioactivity values    
        molar =i*(10**-9) # Converts nM to M
        activity.append(-np.log10(molar))
            
    dataframe['standard_value'] = activity
    
    return dataframe

def outlier_removal_IQR(dataframe):
    """
    Remove outliers using the inter quartile range technique
    """
    Q1 = dataframe['standard_value'].quantile(0.25)
    Q3 = dataframe['standard_value'].quantile(0.75)
    IQR = Q3 - Q1    #IQR is interquartile range. 

    filter_ = (dataframe['standard_value'] >= Q1 - 1.5 * IQR) & (dataframe['standard_value'] <= Q3 + 1.5 *IQR)
    dataframe = dataframe.loc[filter_]  
    
    return dataframe
    
def interest_columns(dataframe):
    """
    Returns a dataframe that contains only the interest columns
    """
    df_dataset = dataframe.filter(['molecule_chembl_id', 'canonical_smiles', 'standard_value'], axis=1)
    return df_dataset
    