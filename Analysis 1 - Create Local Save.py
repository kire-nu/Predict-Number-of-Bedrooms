# -*- coding: utf-8 -*-
"""
@author: Erik Olofsson

Script to create local save of Data Set
Used to save run time in reading and used for analysis
"""

import json
import urllib.request

#Read data from API
propertySampleObjects = []
with urllib.request.urlopen("https://storage.googleapis.com/street_group_data_science/street_group_data_science_bedrooms_test.json") as url:
    for line in url:
        propertySampleObjects.append(json.loads(line))

#Save locally for run time purposes
with open('PropertySampleObjects.json', 'w') as f:
    json.dump(propertySampleObjects, f)

