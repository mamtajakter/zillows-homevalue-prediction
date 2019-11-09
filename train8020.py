
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
import os
import random
# Going to use these 5 base models for the stacking
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.cross_validation import KFold;

# Input data files are available in the "inputs/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# print(os.listdir("inputs"))
# print(os.listdir("inputs/zillow-prize-1"))
# print(os.listdir("inputs/zillowfinaldataset"))

merged2016 = pd.read_csv('inputs/df16_Final_NoNull_meregedNewFeatures.csv', low_memory=False)

train_soja_index = merged2016.sample( int(len(merged2016) * 0.8) ).index
test_soja_index = merged2016[merged2016.index.isin(train_soja_index)==False].index
print(train_soja_index)
print(test_soja_index)

merged2016.shape

# -------------------------------------------------------------------------------------------------

warnings.filterwarnings('ignore')

merged2016.columns

# -------------------------------------------------------------------------------------------------

# Store our passenger ID for easy access
# ParcelID = merged2016['id_parcel']

df_train = merged2016[merged2016.index.isin(train_soja_index)]
df_test = merged2016[merged2016.index.isin(test_soja_index)]


df_test[['id_parcel', 'logerror']].to_csv('test8020_id_label.csv', index=False)



ParcelID_train  = df_train.id_parcel
ParcelID_test  = df_test.id_parcel

print (df_train.shape)
print (df_test.shape)


# Split the merged set into the training set and labels
y_train = df_train['logerror']
x_train = df_train #.drop(['logerror'], axis=1)

# print(y_train.shape)
# print(y_train.head(10))
# print(x_train.shape)
# print(x_train.columns)

# -------------------------------------------------------------------------------------------------

x_train['year'], x_train['month'], x_train['day'] = x_train['transactiondate'].str.split('-').str
x_train = x_train.drop(['transactiondate', 'logerror'], axis=1)
x_train.dtypes

# -------------------------------------------------------------------------------------------------

#config
# colormap = plt.cm.viridis
# plt.figure(figsize=(20,20))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)

# #create the correlation map
# corr = x_train.astype(float).corr()

# #create a mask for the null values
# mask = corr.isnull()

# #plot the heatmap
# sns.heatmap(corr, mask=mask, linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

# -------------------------------------------------------------------------------------------------

# for x in x_train.columns:
# 	if x not in df_test.columns:
# 		print (x + ' is not!!!!!')

# -------------------------------------------------------------------------------------------------

x_train = x_train.drop(['id_parcel'], axis=1)

# Convert training set and labels to Numpy array
x_train = x_train.values
y_train = y_train.values

# -------------------------------------------------------------------------------------------------




# here if we have a 3m dataset with the engineered features and additional features w should use. But I don't see one shared
df_test = df_test.drop(['id_parcel', 'logerror', 'transactiondate'], axis=1)
df_test.columns
# print(len(df_test))
# -------------------------------------------------------------------------------------------------

def get_random_day():
    global df_test
    # print (len(df_test) , ' from inside')
    return np.random.randint(1,30,len(df_test))


x_test_201610 = df_test.copy()
x_test_201610['year'], x_test_201610['month'], x_test_201610['day'] = [2016,10,get_random_day()]
#x_test_201610.to_csv("x_df_test.copy()_201610.csv")
# print (x_test_201610.isnull().sum())

x_test_201610 = x_test_201610.values

x_test_201611 = df_test.copy()
x_test_201611['year'], x_test_201611['month'], x_test_201611['day'] = [2016,11,get_random_day()]
x_test_201611 = x_test_201611.values

x_test_201612 = df_test.copy()
x_test_201612['year'], x_test_201612['month'], x_test_201612['day'] = [2016,12,get_random_day()]
x_test_201612 = x_test_201612.values

x_test_201710 = df_test.copy()
x_test_201710['year'], x_test_201710['month'], x_test_201710['day'] = [2017,10,get_random_day()]
x_test_201710 = x_test_201710.values

x_test_201711 = df_test.copy()
x_test_201711['year'], x_test_201711['month'], x_test_201711['day'] = [2017,11,get_random_day()]
x_test_201711 = x_test_201711.values

x_test_201712 = df_test.copy()
x_test_201712['year'], x_test_201712['month'], x_test_201712['day'] = [2017,12,get_random_day()]
#x_test_201712.to_csv("x_df_test_201712.csv")
x_test_201712 = x_test_201712.values

print("test sets populated with random transaction dates")

# -------------------------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test_201610 = scaler.transform(x_test_201610)
x_test_201611 = scaler.transform(x_test_201611)
x_test_201612 = scaler.transform(x_test_201612)
x_test_201710 = scaler.transform(x_test_201710)
x_test_201711 = scaler.transform(x_test_201711)
x_test_201712 = scaler.transform(x_test_201712)

# -------------------------------------------------------------------------------------------------

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        return(self.clf.fit(x,y).feature_importances_)

# -------------------------------------------------------------------------------------------------

def get_oof(clf, x_train, y_train, x_test_201610, x_test_201611, x_test_201612, x_test_201710, x_test_201711, x_test_201712):
    oof_train = np.zeros((ntrain,))
    
    oof_test_201610 = np.zeros((ntest,))
    oof_test_201611 = np.zeros((ntest,))
    oof_test_201612 = np.zeros((ntest,))
    oof_test_201710 = np.zeros((ntest,))    
    oof_test_201711 = np.zeros((ntest,))
    oof_test_201712 = np.zeros((ntest,))
    
    oof_test_skf_201610 = np.empty((NFOLDS, ntest))
    oof_test_skf_201611 = np.empty((NFOLDS, ntest))
    oof_test_skf_201612 = np.empty((NFOLDS, ntest))
    oof_test_skf_201710 = np.empty((NFOLDS, ntest))
    oof_test_skf_201711 = np.empty((NFOLDS, ntest))
    oof_test_skf_201712 = np.empty((NFOLDS, ntest))
    
    #train_index: indicies of training set
    #test_index: indicies of testing set
     
    for i, (train_index, test_index) in enumerate(kf):
        #break the dataset down into two sets, train and test
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        
        clf.train(x_tr, y_tr)
        
        #make a predition on the test data subset
        oof_train[test_index] = clf.predict(x_te)
        
        #use the model trained on the first fold to make a prediction on the entire test data 
        oof_test_skf_201610[i, :] = clf.predict(x_test_201610)
        oof_test_skf_201611[i, :] = clf.predict(x_test_201611)
        oof_test_skf_201612[i, :] = clf.predict(x_test_201612)
        oof_test_skf_201710[i, :] = clf.predict(x_test_201710)
        oof_test_skf_201711[i, :] = clf.predict(x_test_201711)
        oof_test_skf_201712[i, :] = clf.predict(x_test_201712)
    
    #take an average of all of the folds
    oof_test_201610[:] = oof_test_skf_201610.mean(axis=0)
    oof_test_201611[:] = oof_test_skf_201611.mean(axis=0)
    oof_test_201612[:] = oof_test_skf_201612.mean(axis=0)
    oof_test_201710[:] = oof_test_skf_201710.mean(axis=0)
    oof_test_201711[:] = oof_test_skf_201711.mean(axis=0)
    oof_test_201712[:] = oof_test_skf_201712.mean(axis=0)
    
    return oof_train.reshape(-1, 1), oof_test_201610.reshape(-1, 1), oof_test_201611.reshape(-1, 1), oof_test_201612.reshape(-1, 1), oof_test_201710.reshape(-1, 1), oof_test_201711.reshape(-1, 1), oof_test_201712.reshape(-1, 1)

# -------------------------------------------------------------------------------------------------

SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction

# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 400,
    'learning_rate' : 0.75
}

# Support Vector Classifier parameters 
svm_params = {
    'C' : 0.025,
    'epsilon':0.1
    }

# Gradient Boosting parameters
gb_regressor_params = {
    'n_estimators':500, 
    'learning_rate':0.1,
    'max_depth':1, 
    'random_state':0, 
    'loss':'ls'
}

# -------------------------------------------------------------------------------------------------

### Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostRegressor, seed=SEED, params=ada_params)
gb_regressor = SklearnHelper(clf=GradientBoostingRegressor, seed=SEED, params=gb_regressor_params)
svm = SklearnHelper(clf=LinearSVR, seed=SEED, params=svm_params)

# -------------------------------------------------------------------------------------------------


# here where you can notice our result are different from him, because we don't have full 3m records with eng. features

ntrain = x_train.shape[0]
print(ntrain)
ntest = x_test_201610.shape[0] #need the size of a test set
print(ntest)

kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# -------------------------------------------------------------------------------------------------

svm_oof_train, svm_oof_test_201610, svm_oof_test_201611, svm_oof_test_201612, svm_oof_test_201710, svm_oof_test_201711, svm_oof_test_201712 = get_oof(svm,x_train, y_train, x_test_201610, x_test_201611, x_test_201612, x_test_201710, x_test_201711, x_test_201712) # Support Vector Classifier
print("SVM Training is complete")

# -------------------------------------------------------------------------------------------------

et_oof_train, et_oof_test_201610, et_oof_test_201611, et_oof_test_201612, et_oof_test_201710, et_oof_test_201711, et_oof_test_201712 = get_oof(et, x_train, y_train, x_test_201610, x_test_201611, x_test_201612, x_test_201710, x_test_201711, x_test_201712) # Extra Trees
print("Extra Trees Regressor Training is complete")

# -------------------------------------------------------------------------------------------------

rf_oof_train, rf_oof_test_201610, rf_oof_test_201611, rf_oof_test_201612, rf_oof_test_201710, rf_oof_test_201711, rf_oof_test_201712 = get_oof(rf,x_train, y_train, x_test_201610, x_test_201611, x_test_201612, x_test_201710, x_test_201711, x_test_201712) # Random Forest
print("Random Forest Regressor Training is complete")

# -------------------------------------------------------------------------------------------------

ada_oof_train, ada_oof_test_201610, ada_oof_test_201611, ada_oof_test_201612, ada_oof_test_201710, ada_oof_test_201711, ada_oof_test_201712 = get_oof(ada, x_train, y_train, x_test_201610, x_test_201611, x_test_201612, x_test_201710, x_test_201711, x_test_201712) # AdaBoost 
print("Ada Boost Regressor Training is complete")

# -------------------------------------------------------------------------------------------------

gb_regressor_oof_train, gb_regressor_oof_test_201610, gb_regressor_oof_test_201611, gb_regressor_oof_test_201612, gb_regressor_oof_test_201710, gb_regressor_oof_test_201711, gb_regressor_oof_test_201712 = get_oof(gb_regressor,x_train,y_train,x_test_201610, x_test_201611, x_test_201612, x_test_201710, x_test_201711, x_test_201712)
print("Gradient Boost Regressor Training is complete")

print("Training is complete")

# -------------------------------------------------------------------------------------------------

rf_feature = rf.feature_importances(x_train,y_train)
print("rf_feature", rf_feature)
et_feature = et.feature_importances(x_train, y_train)
print("et_feature", et_feature)
ada_feature = ada.feature_importances(x_train, y_train)
print("ada_feature", ada_feature)
gb_regressor_feature = gb_regressor.feature_importances(x_train,y_train)
print("gb_regressor_feature", gb_regressor_feature)

# -------------------------------------------------------------------------------------------------

########### IMPORTANT ##################################
# copy the results from the previouse step, 
# Change the name by adding 's' at the end of each variable 
# and make sure the commas are there, here then run 
rf_features =[0.00712077, 0.0154129 , 0.01328543, 0.00491346, 0.05410754,
       0.04719976, 0.00152633, 0.0039276 , 0.01082016, 0.00366175,
       0.0560444 , 0.02891938, 0.03644783, 0.00208835, 0.00277777,
       0.01168592, 0.02540691, 0.02104612, 0.0014729 , 0.03108774,
       0.01324667, 0.0036179 , 0.00518589, 0.02510051, 0.00295117,
       0.04358489, 0.05758153, 0.0        , 0.05086191, 0.07083895,
       0.02609021, 0.00841823, 0.00658086, 0.05187097, 0.00180535,
       0.01945536, 0.01197479, 0.02241662, 0.03085848, 0.03146608,
       0.09224914, 0.0        , 0.01207322, 0.03281829]
et_features =[0.01606823, 0.02266831, 0.02306847, 0.00974802, 0.03781387,
       0.04317743, 0.00657941, 0.00413675, 0.00446569, 0.01862182,
       0.03216083, 0.02238427, 0.02898708, 0.01343336, 0.00523533,
       0.0180944 , 0.01513021, 0.05620538, 0.00262948, 0.02875986,
       0.0112263 , 0.01165052, 0.00783428, 0.03263333, 0.00951287,
       0.06711767, 0.03605616, 0.0        , 0.02878938, 0.06624791,
       0.01264109, 0.01339027, 0.00309201, 0.02914838, 0.00131685,
       0.02591188, 0.01376184, 0.0443393 , 0.03945813, 0.0158159 ,
       0.04421989, 0.0        , 0.02805766, 0.04841022]
ada_features =[4.12735032e-04, 2.45416936e-03, 3.06445122e-03, 1.37801851e-04,
       6.18776354e-02, 8.46919398e-03, 5.75572084e-06, 0.00000000e+00,
       3.21891531e-03, 8.18067331e-03, 3.15549761e-02, 1.63272677e-02,
       2.85314907e-02, 1.71683948e-03, 1.45392010e-03, 2.28687001e-03,
       3.73208903e-02, 6.28285959e-02, 3.53946738e-03, 8.43582672e-02,
       8.25738782e-04, 1.56493335e-03, 2.39808486e-05, 4.07387131e-02,
       3.50817714e-04, 8.61170924e-02, 1.22630784e-02, 0.00000000e+00,
       2.00606765e-02, 1.27635521e-02, 4.12605447e-02, 5.36086102e-03,
       3.13695148e-02, 2.00764002e-02, 0.00000000e+00, 1.32184119e-02,
       5.21629473e-02, 3.08163208e-02, 3.85701206e-02, 1.44922962e-01,
       4.21006114e-02, 0.00000000e+00, 3.60455524e-02, 1.16472538e-02]
gb_regressor_features = [0.0   , 0.018, 0.004, 0.0   , 0.084, 0.04 , 0.0   , 0.002, 0.006,
       0.014, 0.136, 0.0   , 0.006, 0.02 , 0.0   , 0.0   , 0.0   , 0.008,
       0.0   , 0.024, 0.008, 0.0   , 0.0   , 0.002, 0.0   , 0.144, 0.07 ,
       0.0   , 0.08 , 0.076, 0.0   , 0.0   , 0.0   , 0.022, 0.0   , 0.06 ,
       0.0   , 0.002, 0.006, 0.048, 0.104, 0.0   , 0.016, 0.0   ]

# -------------------------------------------------------------------------------------------------

cols = df_train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( { 
    'features': cols,
    'Random Forest feature importances': rf_features,
    'Extra Trees  feature importances': et_features,
    'AdaBoost feature importances': ada_features,
    'Gradient Regressor feature importances': gb_regressor_features
    })

feature_dataframe.head(3)

# -------------------------------------------------------------------------------------------------

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Random Forest feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest feature importances',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
# py.iplot(fig)


# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Extra Trees  feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Extra Trees  feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Extra Trees  feature importances',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
# py.iplot(fig)

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'AdaBoost feature importances',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
# py.iplot(fig)



# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Gradient Regressor feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Gradient Regressor feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Gradient Regressor Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
# py.iplot(fig)

# -------------------------------------------------------------------------------------------------

feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise
feature_dataframe.head(50)

# -------------------------------------------------------------------------------------------------

y = feature_dataframe['mean'].values
x = feature_dataframe['features'].values
data = [go.Bar(
            x= x,
             y= y,
            width = 0.5,
            marker=dict(
               color = feature_dataframe['mean'].values,
            colorscale='Portland',
            showscale=True,
            reversescale = False
            ),
            opacity=0.6
        )]

layout= go.Layout(
    autosize= True,
    title= 'Barplots of Mean Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
# py.iplot(fig, filename='bar-direct-labels')

# -------------------------------------------------------------------------------------------------

#predictions from first layer become data input for second layer
base_predictions_train = pd.DataFrame( 
    {
    'RandomForest': rf_oof_train.ravel(),
    'ExtraTrees': et_oof_train.ravel(),
    'AdaBoost': ada_oof_train.ravel(),
    'GradientRegressor': gb_regressor_oof_train.ravel()
    })
#predictions for all instances in the training set
base_predictions_train.head(3)

# -------------------------------------------------------------------------------------------------

data = [
    go.Heatmap(
        z= base_predictions_train.astype(float).corr().values ,
        x=base_predictions_train.columns.values,
        y= base_predictions_train.columns.values,
          colorscale='Portland',
            showscale=True,
            reversescale = True
    )
]
# py.iplot(data, filename='labelled-heatmap')

# -------------------------------------------------------------------------------------------------

x_train = np.concatenate(( et_oof_train, rf_oof_train, gb_regressor_oof_train, svm_oof_train), axis=1)

x_test_201610 = np.concatenate(( et_oof_test_201610, rf_oof_test_201610, gb_regressor_oof_test_201610, ada_oof_test_201610), axis=1)
x_test_201611 = np.concatenate(( et_oof_test_201611, rf_oof_test_201611, gb_regressor_oof_test_201611, ada_oof_test_201611), axis=1)
x_test_201612 = np.concatenate(( et_oof_test_201612, rf_oof_test_201612, gb_regressor_oof_test_201612, ada_oof_test_201612), axis=1)
x_test_201710 = np.concatenate(( et_oof_test_201710, rf_oof_test_201710, gb_regressor_oof_test_201710, ada_oof_test_201710), axis=1)
x_test_201711 = np.concatenate(( et_oof_test_201711, rf_oof_test_201711, gb_regressor_oof_test_201711, ada_oof_test_201711), axis=1)
x_test_201712 = np.concatenate(( et_oof_test_201712, rf_oof_test_201712, gb_regressor_oof_test_201712, ada_oof_test_201712), axis=1)

# -------------------------------------------------------------------------------------------------

gbm = xgb.XGBRegressor(
    learning_rate = 0.02,
n_estimators= 2000,
max_depth= 4,
min_child_weight= 2,
#gamma=1,
gamma=0.9,                        
subsample=0.8,
colsample_bytree=0.8,
objective= 'reg:linear',
nthread= -1,
scale_pos_weight=1
).fit(x_train, y_train)

# -------------------------------------------------------------------------------------------------

#generate the predictions
predictions_201610 = gbm.predict(x_test_201610).round(4)
predictions_201611 = gbm.predict(x_test_201611).round(4)
predictions_201612 = gbm.predict(x_test_201612).round(4)

predictions_201710 = gbm.predict(x_test_201710).round(4)
predictions_201711 = gbm.predict(x_test_201711).round(4)
predictions_201712 = gbm.predict(x_test_201712).round(4)

# -------------------------------------------------------------------------------------------------

StackingSubmission = pd.DataFrame({ '201610': predictions_201610, 
                                         '201611': predictions_201611,
                                         '201612': predictions_201612,
                                         '201710': predictions_201710,
                                         '201711': predictions_201711,
                                         '201712': predictions_201712,
                                         'ParcelId': ParcelID_test,
                            })
print(StackingSubmission)
print('Writing csv ...')
StackingSubmission.to_csv('ensemble_regressor_final_8020.csv', index=False, float_format='%.4f')

