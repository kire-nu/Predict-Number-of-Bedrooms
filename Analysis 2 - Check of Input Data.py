# -*- coding: utf-8 -*-
"""
@author: Erik Olofsson

Check of input data
"""

import json
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import preprocessing
from matplotlib import pyplot
from sklearn.metrics import r2_score


# Read from local save
propertySampleObjects = json.load(open('PropertySampleObjects.json'))
    
# Store as data frame
propertySampleDF = pd.DataFrame(propertySampleObjects)

# Check data types
print(propertySampleDF.dtypes)
# Not all data types are numeric

# Convert to numeric
propertySampleDF["number_habitable_rooms"] = pd.to_numeric(propertySampleDF.number_habitable_rooms, errors='coerce')
propertySampleDF["number_heated_rooms"] = pd.to_numeric(propertySampleDF.number_heated_rooms, errors='coerce')
propertySampleDF["estimated_min_price"] = pd.to_numeric(propertySampleDF.estimated_min_price, errors='coerce')
propertySampleDF["estimated_max_price"] = pd.to_numeric(propertySampleDF.estimated_max_price, errors='coerce')
propertySampleDF["bedrooms"] = pd.to_numeric(propertySampleDF.bedrooms, errors='coerce')

# Check Data types again, should all be numeric
print(propertySampleDF.dtypes)

# Save as local copy to use later
propertySampleDF.to_hdf('PropertySampleDF.h5', key='df', mode='w')

# Change property_type to numerical and remove property_type
enc = preprocessing.OrdinalEncoder()
propertyTypeEncoded = enc.fit_transform(propertySampleDF[['property_type']])
propertySampleDF['property_type_enc'] = pd.DataFrame(propertyTypeEncoded)
propertySampleDF = propertySampleDF[['total_floor_area','number_habitable_rooms',
                                     'number_heated_rooms','estimated_min_price','estimated_max_price',
                                     'longitude','latitude','bedrooms','property_type_enc']]

# Summarize the data set
pd.set_option('display.max_columns', 500)
print(propertySampleDF.shape)
print(propertySampleDF.head(20))
print(propertySampleDF.describe())
print(propertySampleDF.groupby('bedrooms').size())
# Looking at the .describe heated and habitable rooms look similar, could there be a correlation?
plt.scatter(propertySampleDF['number_habitable_rooms'],propertySampleDF['number_heated_rooms'])
z = numpy.polyfit(propertySampleDF['number_habitable_rooms'], propertySampleDF['number_heated_rooms'], 1)
y_hat = numpy.poly1d(z)(propertySampleDF['number_habitable_rooms'])
plt.plot(propertySampleDF['number_habitable_rooms'], y_hat, "r--", lw=1)
text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(propertySampleDF['number_heated_rooms'],y_hat):0.3f}$"
plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
     fontsize=14, verticalalignment='top')
plt.xlabel('number_habitable_rooms')
plt.ylabel('number_heated_rooms')
plt.show()
# There is some correlation, but not fully. There are some that have many habitable rooms but less heated. 
# So which is best for bedrooms

# Plot rooms against bedrooms
plt.scatter(propertySampleDF['number_habitable_rooms'],propertySampleDF['bedrooms'])
z = numpy.polyfit(propertySampleDF['number_habitable_rooms'], propertySampleDF['bedrooms'], 1)
y_hat = numpy.poly1d(z)(propertySampleDF['number_habitable_rooms'])
plt.plot(propertySampleDF['number_habitable_rooms'], y_hat, "r--", lw=1)
text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(propertySampleDF['bedrooms'],y_hat):0.3f}$"
plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
     fontsize=14, verticalalignment='top')
plt.xlabel('number_habitable_rooms')
plt.ylabel('bedrooms')
plt.show()

plt.scatter(propertySampleDF['number_heated_rooms'],propertySampleDF['bedrooms'])
z = numpy.polyfit(propertySampleDF['number_heated_rooms'], propertySampleDF['bedrooms'], 1)
y_hat = numpy.poly1d(z)(propertySampleDF['number_heated_rooms'])
plt.plot(propertySampleDF['number_heated_rooms'], y_hat, "r--", lw=1)
text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(propertySampleDF['bedrooms'],y_hat):0.3f}$"
plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
     fontsize=14, verticalalignment='top')
plt.xlabel('number_heated_rooms')
plt.ylabel('bedrooms')
plt.show()
# Both plots shows little correlation between rooms and bedrooms. 
# There are instances with 0 rooms but wit multiple bedrooms. Suggests issues with data set

# Check if there bedrooms > rooms
propertySampleRoomsCheckDF = propertySampleDF.loc[(propertySampleDF['number_habitable_rooms'] < propertySampleDF['bedrooms']) | 
                                                  (propertySampleDF['number_heated_rooms'] < propertySampleDF['bedrooms'])]
print(propertySampleRoomsCheckDF.shape)
print(propertySampleRoomsCheckDF.head(20))
# This confirms that there are 17117 entries with more bedrooms than rooms.
# This should not be possible, thus issues with the data set
# ideally, this should be cleared from data set to make predictions

# Are max and min price correlated?
plt.scatter(propertySampleDF['estimated_min_price'],propertySampleDF['estimated_max_price'])
z = numpy.polyfit(propertySampleDF['estimated_min_price'], propertySampleDF['estimated_max_price'], 1)
y_hat = numpy.poly1d(z)(propertySampleDF['estimated_min_price'])
plt.plot(propertySampleDF['estimated_min_price'], y_hat, "r--", lw=1)
text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(propertySampleDF['estimated_max_price'],y_hat):0.3f}$"
plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
     fontsize=14, verticalalignment='top')
plt.xlabel('estimated_min_price')
plt.ylabel('estimated_max_price')
plt.show()
# Yes, they are very correlated with an R2 value of .98

# There are some properties with very large floor are, what is the relationship with bedrooms?
plt.scatter(propertySampleDF['total_floor_area'],propertySampleDF['bedrooms'])
z = numpy.polyfit(propertySampleDF['total_floor_area'], propertySampleDF['bedrooms'], 1)
y_hat = numpy.poly1d(z)(propertySampleDF['total_floor_area'])
plt.plot(propertySampleDF['total_floor_area'], y_hat, "r--", lw=1)
text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(propertySampleDF['bedrooms'],y_hat):0.3f}$"
plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes,
     fontsize=14, verticalalignment='top')
plt.xlabel('total_floor_area')
plt.ylabel('bedrooms')
plt.ylim(ymax = 20, ymin = 0)
plt.show()
# There is only one real outlier. But also shows that there is little correlation between floor ared and bedrooms

# Since we have shown that min and max price is correlated, we only use one. 
# Rooms also overlap to a large extent and we have shown that there are issues with it.
# So we'll only use one set of rooms, habitable as it has slighlty better R2
# Location is important, but for visualisation of data
propertySampleDF = propertySampleDF[['total_floor_area','number_habitable_rooms',
                                     'estimated_min_price','bedrooms','property_type_enc']]

# Data Visualization
propertySampleDF.plot(kind='box', subplots=True, sharex=False, sharey=False)
pyplot.show()
propertySampleDF.hist()
pyplot.show()
scatter_matrix(propertySampleDF)
pyplot.show()
# These graphs do not tell too much of interest that has not already been found
# The main thing of interest is that the properties with large number of rooms are not in particular expensive













