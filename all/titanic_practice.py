
# coding: utf-8

# In[188]:

#import libraries needed for this project
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# In[24]:
#import the training data set
trainData = pd.read_csv("train.csv")
# In[25]:
# fill missing data with average AGE
trainData1 = trainData.copy()
trainData1.loc[trainData1["Age"].isnull(),"Age"]= trainData1.Age.mean()


# In[53]:
trainingDataSet = trainData1.loc[: , trainData1.columns !="Survived"]
# In[49]:
#extract the "Survived" Column
testingDataSet = trainData1["Survived"]


# In[50]:
#split data into training and testing data set
X_train, X_test, Y_train, Y_test = train_test_split(trainingDataSet, testingDataSet, random_state = 0)


# In[57]:
lgr = LogisticRegression()
Pclass = X_train["Pclass"].values.reshape(-1,1)
model = lgr.fit(Pclass, Y_train)


# In[58]:
testing_Pclass = X_test["Pclass"].values.reshape(-1,1)
prediction = model.predict(testing_Pclass)


# In[65]:
model.score(testing_Pclass, Y_test)
# In[68]:
metrics.confusion_matrix(Y_test, prediction)


# In[88]:
Pclass_gender =X_train[["Pclass","Sex"]]
# In[90]:
Pclass_gender.Sex = pd.factorize(Pclass_gender.Sex)[0]
# In[92]:
Pclass_gender = np.array(Pclass_gender)

# In[94]:
model2 = lgr.fit(Pclass_gender, Y_train)
# In[98]:
testing_Pclass_gender = X_test[["Pclass","Sex"]]
testing_Pclass_gender.Sex = pd.factorize(testing_Pclass_gender.Sex)[0]
# In[100]:
model2.score(testing_Pclass_gender, Y_test)
# In[228]:
#create a model using decision tree
Decison_tree = DecisionTreeClassifier(criterion="entropy", random_state=200, max_depth=5)
# In[125]:
#create a decision tree model on the date set
model3 = Decison_tree.fit(Pclass_gender, Y_train)
# In[126]:

#prediction
prediction3 = model3.predict(testing_Pclass_gender)
# In[127]:
#display the model score 
model3.score(testing_Pclass_gender, Y_test)
# In[128]:
#display as a metric
metrics.confusion_matrix(Y_test, prediction3)
# In[174]:

#import the testing data set
prediction_data = pd.read_csv("test.csv")
# In[175]:
#create a copy of the data set 
prediction_data2 =prediction_data.copy()
# In[176]:
prediction_data2 = prediction_data2[["Pclass","Sex"]]
# In[177]:

#factorize the "sex" column 
prediction_data2.Sex = pd.factorize(prediction_data2.Sex)[0]
# In[178]:
prediction_data2 = np.array(prediction_data2)

# In[179]:
model3.predict(prediction_data2)
# In[180]:
prediction4 = prediction_data["PassengerId"]
# In[165]:
prediction_data3 = pd.DataFrame(prediction_data2)
# In[168]:

# In[183]:

#convert it into data frame
prediction4 = pd.DataFrame(prediction4)
# In[184]:
#create a new column
prediction4["Survived"] = prediction_data3[1]
# In[187]:

#output the data set as a csv file
prediction4.to_csv("Submission.csv", index=False)

