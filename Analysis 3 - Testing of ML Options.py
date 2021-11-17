# -*- coding: utf-8 -*-
"""
@author: Erik Olofsson
"""

import pandas as pd
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



def TestModel(X, y, RunSVC):
    # Function to test ML options. SVC has long run times, so added toggle for it
    # Input parameters
    # X, y, Data
    # RunSVC: Boolean for running SVC
    
    #Seperate training and validation sets
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    # Spot check algorithms
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    if (RunSVC == True):
        models.append(('SVM', SVC(gamma='auto')))
        
    # Evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)    
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        print('%s, CVS: %f (%f), Accuracy: %f' % 
              (name, cv_results.mean(), cv_results.std(), accuracy_score(Y_validation, predictions)))

    # Compare Algorithms
    pyplot.boxplot(results, labels=names)
    pyplot.title('Algorithm Comparison')
    pyplot.show()



# Load local dataframe saved in earler stage
propertySampleDF = pd.read_hdf('PropertySampleDF.h5', key='df')

# Change property_type to numerical and remove property_type
enc = preprocessing.OrdinalEncoder()
propertyTypeEncoded = enc.fit_transform(propertySampleDF[['property_type']])
propertySampleDF['property_type_enc'] = pd.DataFrame(propertyTypeEncoded)
propertySampleDF = propertySampleDF[['total_floor_area','number_habitable_rooms',
                                     'number_heated_rooms','estimated_min_price','estimated_max_price',
                                     'longitude','latitude','bedrooms','property_type_enc']]

# y is always bedrooms
y = propertySampleDF['bedrooms'].to_numpy()

# Previous analysis showed we dont need number_heated_rooms and estimated_max_price, thus remove these
X = propertySampleDF[['total_floor_area','number_habitable_rooms','estimated_min_price',
                      'longitude','latitude','property_type_enc']]

# Test ML algoritms with these columns
TestModel(X, y, False)
# Best: LDA, CVS: 0.684609 (0.001609), Accuracy: 0.683885
    
# Location is important, but I expect that it might not add anything in this example
# It might be more suitable with additional data or with satelite images
X = propertySampleDF[['total_floor_area','number_habitable_rooms','estimated_min_price','property_type_enc']]

# Test ML algoritms with these columns
TestModel(X, y, False)
# Best: KNN, CVS: 0.688229 (0.001322), Accuracy: 0.691295
# I.e. somewhat better predictions than with long/lat
    
# Previously shown that there are properties with more bedrooms than rooms (impossible), so remove these from the data set
propertySampleDF = propertySampleDF.loc[(propertySampleDF['number_habitable_rooms'] >= propertySampleDF['bedrooms'])]
X = propertySampleDF[['total_floor_area','number_habitable_rooms','estimated_min_price','property_type_enc']]
y = propertySampleDF['bedrooms'].to_numpy()

# Test ML algoritms with these columns
TestModel(X, y, False)
# Best: LDA, CVS: 0.693566 (0.001660), Accuracy: 0.694521
# I.e. somewhat better still
# There are small differences between LDA and KNN. 

    
# Test all options
#TestModel(X, y, True)
# Due to lack processing power, SVC cannot be run in a reasonable amout of time on my laptop
# Initial testing with smaller dataset showed that SVC resulted in better results than other methods
# However, as I cannot get final results with that method, I have discarded the option
