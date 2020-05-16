#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd

##use algo = decision tree
##sklearn is a package that comes with sidekick learn library : most popular ml lib.
##in this we have a module 'tree' and in that a class 'DecisionTreeClassifier', which implements decision tree algo.
from sklearn.tree import DecisionTreeClassifier

##step 3: spliting the data set in training and testing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

musicdata = pd.read_csv('music.csv')

## we split the data in input and ouput cells
## the age and gender will be input
## the genre of music will be output
X = musicdata.drop(columns=['genre'])
#X
Y = musicdata['genre']
#Y

# ## this func returns a tupple so defining four variables
# ##passing three arguments: i/p, o/p, and argument that specifies size of test dataset,i.e, 20%
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2) 

# ## craeting instance of the taken class
# model = DecisionTreeClassifier()
# model.fit(X_train,Y_train) #this method takes input and output
# #musicdata
# ## now making predictions
# #predictions = model.predict([ [21,1], [22,0] ])
# predictions = model.predict(X_test)
# #predictions

# score = accuracy_score(Y_test,predictions) #the output vary all times it run as this func takes randome values from training and testing
# score

##the output is random because we have less amount of data.

# #now if we have to save the trained data to a file...making accuracy calc in comment form
# from sklearn.externals import joblib
# model = DecisionTreeClassifier()
# model.fit(X,Y)

# joblib.dump(model, 'music-recommender.joblib') #passing 2 args model and the filename you wish

#now if we have to return our trained model...commenting the dump codes
from sklearn.externals import joblib
model = joblib.load('music-recommender.joblib')
predictions = model.predict( [[21,1]] )
predictions


# In[35]:


#visualizing a decision tree
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

musicdata = pd.read_csv('music.csv')
X = musicdata.drop(columns=['genre'])
Y = musicdata['genre']

model = DecisionTreeClassifier()
model.fit(X,Y)

tree.export_graphviz(model, out_file='music-recommender.dot',                    feature_names = ['age', 'gender'],                      class_names = sorted(Y.unique()),                    label = 'all', rounded = True, filled = True)

