# -*- coding: utf-8 -*-
"""
@author: Erik Olofsson
"""

import json
import pandas as pd
import urllib.request
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from joblib import dump

#Read data from API
propertySampleObjects = []
with urllib.request.urlopen("https://storage.googleapis.com/street_group_data_science/street_group_data_science_bedrooms_test.json") as url:
    for line in url:
        propertySampleObjects.append(json.loads(line))


#Store as Dataframe
propertySampleDF = pd.DataFrame(propertySampleObjects)

# Convert to numeric
propertySampleDF["number_habitable_rooms"] = pd.to_numeric(propertySampleDF.number_habitable_rooms, errors='coerce')
propertySampleDF["number_heated_rooms"] = pd.to_numeric(propertySampleDF.number_heated_rooms, errors='coerce')
propertySampleDF["estimated_min_price"] = pd.to_numeric(propertySampleDF.estimated_min_price, errors='coerce')
propertySampleDF["estimated_max_price"] = pd.to_numeric(propertySampleDF.estimated_max_price, errors='coerce')
propertySampleDF["bedrooms"] = pd.to_numeric(propertySampleDF.bedrooms, errors='coerce')

# Change property_type to numerical and remove property_type
enc = preprocessing.OrdinalEncoder()
propertyTypeEncoded = enc.fit_transform(propertySampleDF[['property_type']])
propertySampleDF['property_type_enc'] = pd.DataFrame(propertyTypeEncoded)
propertySampleDF = propertySampleDF[['total_floor_area','number_habitable_rooms',
                                     'number_heated_rooms','estimated_min_price','estimated_max_price',
                                     'longitude','latitude','bedrooms','property_type_enc']]

# Keep only columns and rows required to make good predictions
propertySampleDF = propertySampleDF.loc[(propertySampleDF['number_habitable_rooms'] >= propertySampleDF['bedrooms'])]
X = propertySampleDF[['total_floor_area','number_habitable_rooms','estimated_min_price','property_type_enc']]
y = propertySampleDF['bedrooms'].to_numpy()

# Create Model
model = LinearDiscriminantAnalysis()
model.fit(X, y)

dump(model, 'Model to Predict Number of Bedrooms.joblib') 
