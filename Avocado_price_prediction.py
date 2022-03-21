#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

df= pd.read_csv('avocado.csv') #read avocado.csv

df.head() #get first 5 rows of the data
df.shape #know the shape of the data (rows and columns)
df.info() #information of all the columns
df.describe() #various parameters of the columns
df.isnull().sum() #checking for null values if any

df['Date']=pd.to_datetime(df['Date'])
df['month'] = df['Date'].apply(lambda x:x.month) #extract month and day from Date column
df['day'] = df['Date'].apply(lambda x:x.day)

df['type'] = df['type'].replace(['conventional'], 'non-organic') #replacing conventional type to non-organic
df.drop('Unnamed: 0', inplace = True, axis = 1) #dropping Unnamed column as it is of no use
df.head() #first 5 rows of modified dataframe


#plotting counts of organic and non-organic avocados
ax = df['type'].value_counts().plot(kind = 'bar', figsize=(7,5), title="Counts of Organic vs. Non- Organic")
ax.set_xlabel("Types of avocado")
ax.set_ylabel("Counts")

#plotting average prices of organic and non-organic avocados
sns.boxplot(x = 'type', y = 'AveragePrice', data = df).set(title = "Prices of Organic and Non-Organic including outliers")
plt.show()

#plotting average prices of organic and non-organic avocados over years
sns.boxplot(x = 'year', y = 'AveragePrice', hue = 'type', data = df).set(title="Average prices of Organic  and Non-Organic over years including outliers ")
plt.show()

#ploting histogram of organic prices
grouped = df.groupby('type') #to group organic and non-organic rows
grouped.get_group('organic').hist(figsize = (20,20), grid = True, layout = (4,4), bins = 30)

#ploting histogram of non-organic prices
grouped.get_group('non-organic').hist(figsize = (20,20), grid = True, layout = (4,4), bins = 30)

final_df = df.drop(['region', 'Date'], axis = 1) #dropping region and date columns as they do not define the type of the avocados
label_encoder = preprocessing.LabelEncoder()
final_df['type']= label_encoder.fit_transform(df['type']) #replacing type column by numerical data using label encoding

X = final_df.drop(['type'], axis = 1, inplace = False) #dropping type column to make it as target 
y = final_df['type']

clf = [SVC(), KNeighborsClassifier(), RandomForestClassifier()] #checking accuracy of different classifiers
score = 0
for r_state in range(10,11):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = r_state)
  for c in clf:
    c.fit(X_train,y_train)
    y_pred=c.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    if accuracy>score:
      score=accuracy
      final_state=r_state
      final_classifier=c
print("Maximum accuracy score corresponding to random state ",final_state , "is" ,score, "and classifier is ", final_classifier)

#best performance by RandomForestClassifier
#Performing GridSearch to find best hyperparameters
n_estimators = [100, 300] 
max_depth = [5, 8]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(final_classifier, hyperF, cv = 3, verbose = 1, 
                      n_jobs = 3)
bestF = gridF.fit(X_train, y_train)
print(bestF.best_params_)

#Classification using best hyperparameters
clf = RandomForestClassifier(n_estimators=300, random_state=10, max_depth=8, min_samples_leaf=1, min_samples_split=5)
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

#prediction of type by giving random values of all the colums
input= (1.33, 64236.62, 1036.74, 54454.85,	48.16,	8696.87,	8603.62,	93.25,	0.0, 2015, 12, 27)
input_arr = np.asarray(input)
reshape = input_arr.reshape(1,-1)
prediction = clf.predict(reshape)
print(prediction)
if (prediction[0] == 0):
  print('The type is non-organic')
else:
  print('The type is organic')

#extracting features from data
X = final_df.drop(['AveragePrice'], axis = 1, inplace = False) #dropping AveragePrice column to define it as target
y = final_df['AveragePrice']

#split data into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

#price prediction using LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print('MAE :', metrics.mean_absolute_error(y_test, pred_lr))
print('MSE :', metrics.mean_squared_error(y_test, pred_lr))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, pred_lr)))
print('R2 :', r2_score(y_test, pred_lr)) #coefficient of determination (proportion of variability)

plt.scatter(x = y_test, y = pred_lr)

#price prediction using using Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print('MAE :', metrics.mean_absolute_error(y_test, pred_rf))
print('MSE :', metrics.mean_squared_error(y_test, pred_rf))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, pred_rf)))
print('R2 :', r2_score(y_test, pred_rf))

plt.scatter(x = y_test, y = pred_rf)

#price prediction using using Decision Tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)

print('MAE :', metrics.mean_absolute_error(y_test, pred_dt))
print('MSE :', metrics.mean_squared_error(y_test, pred_dt))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, pred_dt)))
print('R2 :', r2_score(y_test, pred_dt))

plt.scatter(x = y_test, y = pred_dt)

price = pd.DataFrame({'Y-Test' : y_test , 'Pred' : pred_rf}, columns = ['Y-Test', 'Pred'])

sns.lmplot(x = 'Y-Test', y = 'Pred', data = price, palette = 'rainbow')

#plotting region by AveragePrice bar graph
price_ranking=df.groupby('region')[['AveragePrice']].mean().sort_values(by="AveragePrice", ascending=True)

plt.figure(figsize=(20,10))
plt.xticks(rotation=70)

ax = sns.barplot(x=price_ranking.index, y="AveragePrice", data=price_ranking)
ax.set_xlabel('region')
ax.set_ylabel("Average Price")
plt.title('Average Price of Avocado by region')
plt.savefig('price_ranking')