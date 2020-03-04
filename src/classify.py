#!/bin/python3
#Faris Zahrah 2/25/2020


# This serves as an inital classification model for building complaints, 
# with the target feature being whether the violation was enforced or not.
# Can we predict enforcment based on building complaint featueres is the high

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample

import warnings
complaints_df = pd.read_pickle('../pickle/complaints_v2.pkl')
reports_df = pd.read_pickle('../pickle/reports.pkl')


if True:
    warnings.filterwarnings("ignore")


#add a url feature to match with reports_df
complaints_df['url'] = complaints_df.apply(lambda x: x.complaints['url'],axis=1)
reports_df['url'] = reports_df.apply(lambda x: x.complaints['url'],axis=1)


# merge 
df_t = reports_df.merge(complaints_df, on='url')
df_t = df_t.drop_duplicates('url')

df_t = df_t[['direction_x','street_type_x',
                'latitude_x','longitude_x','permits','tanks',
                'neshaps_demolition_notices','holds',
                'complaint_type','street_number_x','enforcement']]


# these are features where a Nan value indicates 0 and all other values indicate 1.
# Making this adjustment now
dict_columns = ['enforcement','permits','tanks','neshaps_demolition_notices','holds']
for i in dict_columns:
    df_t[i] = df_t[i].apply(lambda x: 1 if type(x) == dict else 0 )


# applying a very hacky encodder
categorical = ['direction_x','street_type_x','complaint_type']
le = preprocessing.LabelEncoder()
for c in categorical:
    df_t[c] = df_t[c].apply(str).astype('category')
    #le = preprocessing.LabelEncoder()
    le.fit(df_t[c])
    df_t[c] = le.transform(df_t[c])


#drop rows with missing values, 
df_t = df_t.dropna(axis=0)



#### UPSAMPLING START ######
#We have a major issue regarding sampling
# Separate majority and minority classes
df_majority = df_t[df_t.enforcement==0]
df_minority = df_t[df_t.enforcement==1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=28673,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_t = pd.concat([df_majority, df_minority_upsampled])
 
#### UPSAMPLING STOP ######


# converting object types to floats,ints
categories = {'latitude_x':float,'longitude_x':float,'street_number_x':int}
for c in categories:
    df_t[c] = df_t[c].apply(categories[c])


#seperating features from target
df_t_features = df_t.drop('enforcement',axis=1)
df_t_features.drop('tanks',axis=1,inplace=True)
df_t_target = df_t[['enforcement']]


#train test split, standard operation
X_train, X_test, y_train, y_test =  train_test_split(df_t_features,df_t_target, test_size=0.3)




######## Decision Tree Classifier Start ########

#Simple Decision Tree
dt = DecisionTreeClassifier(max_depth=12,random_state=0)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)

data = {'y_Actual':    list(y_test.enforcement),
        'y_Predicted': predictions
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
conf_mat = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print(f'Confusion Matrix:\n{conf_mat}')
print(f'Decision Tree Classifier Accuracy: {dt.score(X_test,y_test)}')

######## Decision Tree Classifier Stop ########



######## Random Forest Classifier Start ########
clf = RandomForestClassifier(max_depth=12, random_state=0)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

data = {'y_Actual':    list(y_test.enforcement),
        'y_Predicted': prediction
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
#sns.heatmap(confusion_matrix, annot=True)
print(f'\n\nConfusion Matrix:\n{confusion_matrix}')
print(f'Random Forest Classifier Accuracy: {clf.score(X_test,y_test)}')

######## Random Forest Classifier Stop ########




### printing feature importance  start ###
print('\nFeature Importances:')
d = dict(zip(clf.feature_importances_, df_t_features.columns))
for elem in sorted(d.items(),reverse=True) :
    print(f'{elem[1]}: {round(elem[0],6)} ')

### printing feature importance stop ####













