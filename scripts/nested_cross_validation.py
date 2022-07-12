import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

 
# manual nested cross-validation for random forest on a classification dataset

space = {'n_estimators' : [10, 100, 500],
         'max_features' : [2, 4, 6],
         'criterion'    : ['gini', 'entropy']
         }

def nested_cv(internal_x, internal_y, search_space = space, outer_splits = 10, inner_splits = 5):
    """
    Nested Cross Validation is a method that selects Hyper Parameters inside an outer cross validation. 
    This can be used to train the model. This code Stores a model for each outer split and return a list with all these models.
    An external set is left out of cross validation for a less biased evaluation of the models.
    """


    # configure the cross-validation procedure
    cv_outer = StratifiedKFold(n_splits = outer_splits, shuffle=True, random_state=1)

    model_list = list()
    # enumerate splits
    outer_results = list()
    for train_ix, test_ix in cv_outer.split(internal_x, internal_y):
        # split data
        X_train, X_test = internal_x.iloc[train_ix,:], internal_x.iloc[test_ix,:]
        y_train, y_test = np.take(internal_y, train_ix), np.take(internal_y, test_ix)
        
        # configure the cross-validation procedure
        cv_inner = StratifiedKFold(n_splits = inner_splits, shuffle=True, random_state=1)
        
        # define the model
        model = RandomForestClassifier(random_state=1)
        
        # define search
        search = GridSearchCV(model, search_space, scoring='accuracy', cv=cv_inner, refit=True)
        
        # execute search
        result = search.fit(X_train, y_train)
        
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        model_list.append(best_model)
        
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        
        # evaluate the model
        acc = accuracy_score(y_test, yhat)
        
        # store the result
        outer_results.append(acc)
        
        # report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
 
    return model_list


def nested_cv_best(internal_x, internal_y, search_space = space, outer_splits = 10, inner_splits = 5):
    """
    Nested Cross Validation is a method that selects Hyper Parameters inside an outer cross validation. 
    This can be used to train the model. This code returns the model with better evaluation.
    An external set is left out of cross validation for a less biased evaluation of the models.
    """


    # configure the cross-validation procedure
    cv_outer = StratifiedKFold(n_splits = outer_splits, shuffle=True, random_state=1)

    model_list = list()
    # enumerate splits
    outer_results = list()
    for train_ix, test_ix in cv_outer.split(internal_x, internal_y):
        # split data
        X_train, X_test = internal_x.iloc[train_ix,:], internal_x.iloc[test_ix,:]
        y_train, y_test = np.take(internal_y, train_ix), np.take(internal_y, test_ix)
        
        # configure the cross-validation procedure
        cv_inner = StratifiedKFold(n_splits = inner_splits, shuffle=True, random_state=1)
        
        # define the model
        model = RandomForestClassifier(random_state=1)
        
        # define search
        search = GridSearchCV(model, search_space, scoring='accuracy', cv=cv_inner, refit=True)
        
        # execute search
        result = search.fit(X_train, y_train)
        
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        model_list.append(best_model)
        
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        
        # evaluate the model
        acc = accuracy_score(y_test, yhat)
        
        # store the result
        outer_results.append(acc)
        
        # report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
    for i, j in enumerate(outer_results):
        if j == max(outer_results):
            return model_list[i]


def ensemble_predict(list_model, x_test):
    """
    This functions receives a list of models and an array with values to be used in a prediction
    
    The prediction probability will be calculated for each model and the mean of this valeus will be stored in a list
    to be returned as the result of the function
    """
    #empty list to store predictions from each model
    prediction_list = list() 
    
    #Calculates each model activity probability
    for i in list_model:
        prediction = i.predict_proba(x_test) 
        prediction_list.append(prediction)
    
    #calculates model mean probability of activity
    final_vote = list() 
    for i in range(len(prediction)):
        ballot = 0 #variable to be used as a sum of probabilities
        for j in prediction_list:
            ballot += j[i][1]
        fvote = ballot / len(prediction_list) #mean of probabilities for each variable in all available models
        final_vote.append(fvote)
        
    return final_vote
    
def binary_pred(prob_list, t = 0.5):
    """
    Receives a prediction probability list and returns an array with the 1 and 0 results given a fixed threshold for probability
    """
    list_prediction = list()
    for i in prob_list:
        if i >= t:
            list_prediction.append(1)
        else:
            list_prediction.append(0)
    return np.array(list_prediction)