import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
#from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import pickle

#loading Dataset
Master_df = pd.read_csv('Medical_fraud_analysis.csv')

#removing unnecessary columns
Master_df.loc[Master_df['DOD'].isnull(), 'IsDead'] = '0'
Master_df.loc[(Master_df['DOD'].notnull()), 'IsDead'] = '1'
Master_df = Master_df.drop(['DOD'], axis = 1)
Master_df = Master_df.drop(['DOB'], axis = 1)
Master_df = Master_df.drop(['age'], axis = 1) 

#converting date into day-month-year format
Master_df['AdmissionDt'] = pd.to_datetime(Master_df['AdmissionDt'], format = '%Y-%m-%d')
Master_df['DischargeDt'] = pd.to_datetime(Master_df['DischargeDt'], format = '%Y-%m-%d')
Master_df['DaysAdmitted'] = ((Master_df['DischargeDt'] - Master_df['AdmissionDt']).dt.days) + 1
Master_df.loc[Master_df['EncounterType'] == 1, 'DaysAdmitted'] = '0'
Master_df[['EncounterType', 'DaysAdmitted', 'DischargeDt', 'AdmissionDt']].head()
Master_df = Master_df.drop(['DischargeDt'], axis = 1)
Master_df = Master_df.drop(['AdmissionDt'], axis = 1)

#replacing null values with 0
Master_df.loc[Master_df['DeductibleAmtPaid'].isnull(), 'DeductibleAmtPaid'] = '0'

cols= ['ClmAdmitDiagnosisCode', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_10',
       'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
       'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
       'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmProcedureCode_1',
       'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
       'ClmProcedureCode_5', 'ClmProcedureCode_6']

Master_df[cols] = Master_df[cols].replace({np.nan:0})
Master_df

#replacing non-zeroes with 1
for i in cols:
    Master_df[i][Master_df[i] != 0] = 1

#changing datatype of coumns and creating new columns by adding them
Master_df[cols] = Master_df[cols].astype(float)
Master_df['TotalDiagnosis'] = Master_df['ClmDiagnosisCode_1'] + Master_df['ClmDiagnosisCode_10'] + Master_df['ClmDiagnosisCode_2'] + Master_df['ClmDiagnosisCode_3']  + Master_df['ClmDiagnosisCode_4'] + Master_df['ClmDiagnosisCode_5'] + Master_df['ClmDiagnosisCode_6'] + Master_df['ClmDiagnosisCode_7'] + Master_df['ClmDiagnosisCode_8'] + Master_df['ClmDiagnosisCode_9']
Master_df['TotalProcedure'] = Master_df['ClmProcedureCode_1'] + Master_df['ClmProcedureCode_2'] + Master_df['ClmProcedureCode_3'] + Master_df['ClmProcedureCode_4'] + Master_df['ClmProcedureCode_5'] + Master_df['ClmProcedureCode_6']

#removing more unnecessary columns
remove = ['Provider','BeneID', 'ClaimID', 'ClaimStartDt','ClaimEndDt','AttendingPhysician',
           'OperatingPhysician', 'OtherPhysician', 'ClmDiagnosisCode_1',
           'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
           'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
           'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
           'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
           'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6',
           'ClmAdmitDiagnosisCode', 'DeductibleAmtPaid', 'NoOfMonths_PartACov',
           'NoOfMonths_PartBCov', 'DiagnosisGroupCode',
           'State', 'County']
Master_df.drop(columns = remove, axis = 1, inplace = True)

#converting objects values into numeric
Master_df['RenalDiseaseIndicator'] = Master_df['RenalDiseaseIndicator'].replace({'Y':1, '0':0})
Master_df['RenalDiseaseIndicator'] = Master_df['RenalDiseaseIndicator'].astype(int)
Master_df['IsDead'] = Master_df['IsDead'].astype(float)
Master_df['DaysAdmitted'] = Master_df['DaysAdmitted'].astype(float)
Master_df['PotentialFraud'] = Master_df['PotentialFraud'].replace({'Yes':1, 'No':0})
Master_df['PotentialFraud'] = Master_df['PotentialFraud'].astype(int)

#initial train-test splitting 
x = Master_df.drop('PotentialFraud', axis = 1)
y = Master_df.loc[:, 'PotentialFraud']

#saperating numerical data
num_col= ['InscClaimAmtReimbursed', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
       'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Age',
       'DaysAdmitted', 'TotalDiagnosis', 'TotalProcedure']
numerical_columns = x.loc[:, num_col]

#saperating categorical data
cat_col= ['EncounterType', 'Gender', 'Race',
       'RenalDiseaseIndicator', 'ChronicCond_Alzheimer',
       'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
       'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
       'ChronicCond_Depression', 'ChronicCond_Diabetes',
       'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
       'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke','IsDead']
x_cat = x.loc[:, cat_col]

#fit and transform numerical values and then merge them with categorical values
scale = StandardScaler()
x_num = scale.fit_transform(x[num_col])
x_num = pd.DataFrame(x_num, columns = ['InscClaimAmtReimbursed', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
                                       'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Age', 'DaysAdmitted', 
                                       'TotalDiagnosis', 'TotalProcedure'])
x= pd.concat([x_num, x_cat], axis = 1)

#final train-test splitting
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)
acc_score = []

#training model with logistic regression algorithm and finding accuracy acore
lr = LogisticRegression()
lr.fit(x_train, y_train)
train_pred = lr.predict(x_train)
test_pred = lr.predict(x_test)
start = time.time()
lr.score(x_train, y_train)
acc_random_forest = round(accuracy_score(y_test, test_pred) * 100, 2)
f1_random_forest = round(f1_score(y_test, test_pred, average = "binary") * 100, 2)
f_beta_random_forest = round(fbeta_score(y_test, test_pred, average = "binary", beta = 0.5) * 100, 2)
end = time.time()
acc_score.append({'Model':"Logistic Regression", 'Score':accuracy_score(y_train, train_pred), 
                  'Accuracy':accuracy_score(y_test, test_pred), 'Time_Taken':end - start})

#training model with random forest classifier and finding accuracy acore
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
model_score = rfc.predict(x_train)
accuracy = rfc.predict(x_test)
start = time.time()
rfc.score(x_train, y_train)
acc_random_forest = round(accuracy_score(y_test, accuracy) * 100, 2)
f1_random_forest = round(f1_score(y_test, accuracy, average = "binary") * 100, 2)
f_beta_random_forest = round(fbeta_score(y_test, accuracy, average = "binary", beta = 0.5) * 100, 2)
end = time.time()
acc_score.append({'Model':'Random Forest', 'Score':accuracy_score(y_train, model_score), 'Accuracy':accuracy_score(y_test, accuracy), 'Time_Taken':end - start})

#training model with decision tree classifier and finding accuracy acore
dtc =  DecisionTreeClassifier(criterion = 'gini', max_depth = 5, min_samples_split = 2)
dtc.fit(x_train, y_train)
model_score = dtc.predict(x_train)
accuracy = dtc.predict(x_test)
start = time.time()
dtc.score(x_train, y_train)
acc_random_forest = round(accuracy_score(y_test, accuracy) * 100, 2)
f1_random_forest = round(f1_score(y_test, accuracy, average = "binary") * 100, 2)
f_beta_random_forest = round(fbeta_score(y_test, accuracy, average = "binary", beta = 0.5) * 100, 2)
end = time.time()
acc_score.append({'Model':'Decision Tree', 'Score':accuracy_score(y_train, model_score), 'Accuracy':accuracy_score(y_test, accuracy), 'Time_Taken':end - start})

#training model with XG boost classifier and finding accuracy acore
'''xgb = XGBClassifier()
xgb.fit(x_train, y_train)
model_score = xgb.predict(x_train)
accuracy = xgb.predict(x_test)
start = time.time()
xgb.score(x_train, y_train)
acc_random_forest = round(accuracy_score(y_test, accuracy) * 100, 2)
f1_random_forest = round(f1_score(y_test, accuracy, average = "binary") * 100, 2)
f_beta_random_forest = round(fbeta_score(y_test, accuracy, average = "binary", beta = 0.5) * 100, 2)
end = time.time()
acc_score.append({'Model':'XG boost', 'Score':accuracy_score(y_train, model_score), 'Accuracy':accuracy_score(y_test, accuracy), 'Time_Taken':end - start})'''

#training model with gaussian naive-bayes algorithm and finding accuracy acore
gnb = GaussianNB()
gnb.fit(x_train, y_train)
train_pred = gnb.predict(x_train)
test_pred = gnb.predict(x_test)
start = time.time()
gnb.score(x_train, y_train)
acc_random_forest = round(accuracy_score(y_test, test_pred) * 100, 2)
f1_random_forest = round(f1_score(y_test, test_pred, average = "binary") * 100, 2)
f_beta_random_forest = round(fbeta_score(y_test, test_pred, average = "binary", beta = 0.5) * 100, 2)
end = time.time()
acc_score.append({'Model':'Naive Bayes', 'Score':accuracy_score(y_train, train_pred), 'Accuracy':accuracy_score(y_test, test_pred), 'Time_Taken':end- start})

#creating dataframe with model names and their respected accuracy-time
accuracy = pd.DataFrame(acc_score, columns = ['Model', 'Score', 'Accuracy', 'Time_Taken'])
accuracy.sort_values(by = 'Accuracy', ascending = False, inplace = True)
print(accuracy)

'''By inspecting this dataframe it is clear that random forest classifier 
   is the best algorithm for achiving higher accuracy and proper prediction
   Hence, we'll use model trained with random forest classifier for further 
   processing'''

#initialising and dumping 'rfc' model in pickle for prediction
pickle.dump(rfc, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))

#predicting fraud by putting values in selected model
resp = model.predict([[9.542662,2.610838,1.234826,-0.571436,-0.578530,-0.519851,1.832646,0.446318,-0.190910,0,1,1,0,1,1,1,2,2,1,1,1,2,1,1,0.0]])
if(resp == 1):
    print('This claim is potential fraud')
else:
    print('It is not fraudulent claim')