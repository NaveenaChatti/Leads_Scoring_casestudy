# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')
# Importing libraries
import pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Reading datasets
leads = pd.read_csv("Leads.csv")
leads.head()
# Checking the dimensions of the dataframe
leads.shape
# Statistical aspects of the dataframe
leads.describe()
leads.info()
#check for duplicates
sum(leads.duplicated(subset = 'Prospect ID')) == 0
### No duplicate values in Prospect ID

#check for duplicates
sum(leads.duplicated(subset = 'Lead Number')) == 0
### No duplicate values in Lead Number

## EXPLORATORY DATA ANALYSIS

### Data Cleaning & Treatment

# Converting 'Select' values in the data set as null 
leads = leads.replace('Select', np.nan)
# Calculating the percentage of na/null values in the dataset
round(100*leads.isna().sum()/len(leads),2)
#Dropping columns with more than 40% null values

cols=leads.columns

for i in cols:
    if((100*(leads[i].isnull().sum()/len(leads))) >= 40):
        leads.drop(i, 1, inplace = True)
leads.head()
# Again checking null values percentage
round(100*leads.isna().sum()/len(leads),2)
### Data Imputing with Mode values:

#checking value counts of Country column
leads['Country'].value_counts(dropna=False)
# Replacing null values in Country column with mode 'India'
leads['Country'].fillna('India', inplace = True)
#Checking sum of null values on the Country Column
leads['Country'].isna().sum()
#As we can see the Number of Values for India are quite high (nearly 96.9% of the Data)

leads['Country'] = leads['Country'].apply(lambda x: 'India' if x=='India' else 'Outside India')
leads['Country'].value_counts()
#plotting spread of Country columnn 
plt.figure(figsize=(15,5))
p1=sns.countplot(leads.Country, hue=leads.Converted)
p1.set_xticklabels(p1.get_xticklabels(),rotation=90)
plt.show()
#### As we can see the Number of Values for India are quite high (nearly 97% of the Data), this column can be dropped
#creating a list of columns to be droppped

drop_cols=['Country']
#checking value counts of Specialization column
leads['Specialization'].value_counts(dropna=False)
# Replacing null values in the Specialization column with 'Not Specified' since the specialization might not be present in the list
leads['Specialization'].fillna('Not Specified', inplace = True)
#combining Management Specializations because they show similar trends

leads['Specialization'] = leads['Specialization'].replace(['Finance Management','Human Resource Management',
                                                           'Marketing Management','Operations Management',
                                                           'IT Projects Management','Supply Chain Management',
                                                    'Healthcare Management','Hospitality Management',
                                                           'Retail Management'] ,'Management_Specializations')  
#plotting spread of Specialization columnn 
sns.barplot(y='Specialization', x='Converted', palette='husl', data=leads, estimator=np.sum)
#checking value counts of 'What is your current occupation' column
leads['What is your current occupation'].value_counts(dropna=False)
# Filling null values with mode i.e. 'Unemployed'
leads['What is your current occupation'].fillna('Unemployed', inplace = True)
#plotting spread of 'What is your current occupation' columnn 
sns.barplot(y='What is your current occupation', x='Converted', palette='husl', data=leads, estimator=np.sum)
#checking value counts of 'What matters most to you in choosing a course' column
leads['What matters most to you in choosing a course'].value_counts(dropna=False)
# We see many people opt for a course for better career prospects, we replace the null value with the same
leads['What matters most to you in choosing a course'].fillna('Better Career Prospects', inplace = True)
leads['What matters most to you in choosing a course'].isna().sum()
#plotting spread of 'What matters most to you in choosing a course' columnn 
sns.barplot(y='What matters most to you in choosing a course', x='Converted', palette='husl', data=leads, estimator=np.sum)
# we have another Column that is worth Dropping. So we Append to drop_cols List
drop_cols.append('What matters most to you in choosing a course')
drop_cols
#checking value counts of Tags column
leads['Tags'].value_counts(dropna=False)
# Imputing Tags with mode
leads['Tags'].fillna('Not Specified', inplace = True)
leads['Tags'].isnull().sum()
#replacing tags with low frequency with "Other Tags"
leads['Tags'] = leads['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',
                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',
                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',
                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',
                                    'University not recognized','switched off','Already a student','Not doing further education',
                                       'invalid number','wrong number given','Interested  in full time MBA'], 'Other_Tags')
#plotting spread of 'Tags' columnn 
plt.figure(figsize=(15,5))
sns.barplot(y='Tags', x='Converted', palette='husl', data=leads, estimator=np.sum)
#checking value counts of City column
leads['City'].value_counts(dropna=False)
# Replacing na values for city with Maharashtra
leads['City'].fillna('Mumbai', inplace = True)
leads['City'].isnull().sum()
#plotting spread of 'City' columnn 
sns.barplot(y='City', x='Converted', palette='husl', data=leads, estimator=np.sum)
round(100*leads.isna().sum()/len(leads),2)
#checking value counts of Lead Source column
leads['Lead Source'].value_counts(dropna=False)
leads['Lead Source'] = leads['Lead Source'].replace(['bing','Click2call','Press_Release','youtubechannel','welearnblog_Home',
                                                     'WeLearn','blog','Pay per Click Ads','testone','NC_EDM'] ,'Others')

leads['Lead Source'] = leads['Lead Source'].replace('Facebook','Social Media')
#checking value counts of Last Activity column
leads['Last Activity'].value_counts(dropna=False)
leads['Last Activity'] = leads['Last Activity'].replace(['Unreachable','Unsubscribed','Had a Phone Conversation','Approached upfront',
                                                        'View in browser link Clicked', 'Email Marked Spam','Email Received','Resubscribed to emails',
                                                         'Visited Booth in Tradeshow'],'Others')
leads['Lead Source'].fillna('Others', inplace = True)
leads['Last Activity'].fillna('Others', inplace = True)
### Imputing with Median values because the continuous variables have outliers

leads['TotalVisits'].replace(np.NaN, leads['TotalVisits'].median(), inplace =True)
leads['Page Views Per Visit'].replace(np.NaN, leads['Page Views Per Visit'].median(), inplace =True)
### checking null values percentage Post Imputing data
round(100*leads.isna().sum()/len(leads),2)
##checking value counts of Lead Origin column Lead Origin
leads['Lead Origin'].value_counts(dropna=False)
#visualizing count of Lead Origin based on Converted value

plt.figure(figsize=(8,5))
s1=sns.countplot(leads['Lead Origin'], hue=leads.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()
### Inference
API and Landing Page Submission bring higher number of leads as well as conversion.

Lead Add Form has a very high conversion rate but count of leads are not very high.

Lead Import and Quick Add Form get very few leads.
#checking value counts of Lead Source column
leads['Lead Source'].value_counts()
#visualizing count of Lead Source based on Converted value

plt.figure(figsize=(8,5))
s1=sns.countplot(leads['Lead Source'], hue=leads.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()
##visualizing count of Do Not Email & Do Not Call based on Converted value

plt.figure(figsize=(15,5))

ax1=plt.subplot(1, 2, 1)
ax1=sns.countplot(leads['Do Not Call'], hue=leads.Converted)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)

ax2=plt.subplot(1, 2, 2)
ax2=sns.countplot(leads['Do Not Email'], hue=leads.Converted)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
plt.show()
#checking value counts for Do Not Call
leads['Do Not Call'].value_counts(dropna=False)
## We Can append the Do Not Call Column to the list of Columns to be Dropped since > 95% is of only one Value
drop_cols.append('Do Not Call')
drop_cols
#checking value counts for Do Not Email
leads['Do Not Email'].value_counts(dropna=False)
### IMBALANCED VARIABLES THAT CAN BE DROPPED

#checking value counts of Search column
leads.Search.value_counts(dropna=False)
#checking value counts of Newspaper Article column
leads['Newspaper Article'].value_counts(dropna=False)
#checking value counts of X Education Forums column
leads['X Education Forums'].value_counts(dropna=False)
#checking value counts of Magazine column
leads['Magazine'].value_counts(dropna=False)
#checking value counts of Digital Advertisement column
leads['Digital Advertisement'].value_counts(dropna=False)
#checking value counts of Through Recommendations column
leads['Through Recommendations'].value_counts(dropna=False)
#checking value counts of Newspaper column
leads['Newspaper'].value_counts(dropna=False)
#checking value counts of Receive More Updates About Our Courses column
leads['Receive More Updates About Our Courses'].value_counts(dropna=False)
#checking value counts of Update me on Supply Chain Content column
leads['Update me on Supply Chain Content'].value_counts(dropna=False)
#checking value counts of Get updates on DM Content column
leads['Get updates on DM Content'].value_counts(dropna=False)
#checking value counts of I agree to pay the amount through cheque column
leads['I agree to pay the amount through cheque'].value_counts(dropna=False)
#checking value counts of A free copy of Mastering The Interview column
leads['A free copy of Mastering The Interview'].value_counts(dropna=False)
#adding imbalanced columns to the list of columns to be dropped

drop_cols.extend(['Search','Newspaper Article','X Education Forums','Magazine','Digital Advertisement','Through Recommendations',
                     'Newspaper','Receive More Updates About Our Courses','Update me on Supply Chain Content','Get updates on DM Content',
                     'I agree to pay the amount through cheque'])
#checking value counts of last Notable Activity
leads['Last Notable Activity'].value_counts()
#clubbing lower frequency values

leads['Last Notable Activity'] = leads['Last Notable Activity'].replace(['Had a Phone Conversation','Email Marked Spam',
                                                                         'Unreachable','Unsubscribed','Email Bounced',                                                                    
                                                                       'Resubscribed to emails','View in browser link Clicked',
                                                                       'Approached upfront', 'Form Submitted on Website', 
                                                                       'Email Received'],'Other_Notable_activity')
#checking value counts for Last Notable Activity

leads['Last Notable Activity'].value_counts()
#visualizing count of Last Notable Activity based on Converted value

plt.figure(figsize = (14,5))
ax1=sns.countplot(x = "Last Notable Activity", hue = "Converted", data = leads)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
plt.show()
#list of columns to be dropped
drop_cols
#dropping columns
leads = leads.drop(drop_cols,1)
leads.info()
### Outlier Treatment:

#Total Visits
#visualizing spread of variable

plt.figure(figsize=(6,4))
sns.boxplot(y=leads['TotalVisits'])
plt.show()
#checking percentile values for "Total Visits"

leads['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])
#Outlier Treatment: Remove top & bottom 2% of the Column Outlier values

Q3 = leads.TotalVisits.quantile(0.98)
leads = leads[(leads.TotalVisits <= Q3)]
Q1 = leads.TotalVisits.quantile(0.02)
leads = leads[(leads.TotalVisits >= Q1)]
sns.boxplot(y=leads['TotalVisits'])
plt.show()
#Total Time Spent on Website
#visualizing spread of variable

plt.figure(figsize=(6,4))
sns.boxplot(y=leads['Total Time Spent on Website'])
plt.show()
### Since there are no major Outliers for the above variable we don't do any Outlier Treatment for this above Column

#Page Views Per Visit
#visualizing spread of numeric variable

plt.figure(figsize=(6,4))
sns.boxplot(y=leads['Page Views Per Visit'])
plt.show()
#checking spread of "Page Views Per Visit"

leads['Page Views Per Visit'].describe()
#Outlier Treatment: Remove top & bottom 1% of the Column Outlier values

Q3 = leads['Page Views Per Visit'].quantile(0.99)
leads = leads[leads['Page Views Per Visit'] <= Q3]
Q1 = leads['Page Views Per Visit'].quantile(0.01)
leads = leads[leads['Page Views Per Visit'] >= Q1]
sns.boxplot(y=leads['Page Views Per Visit'])
plt.show()
leads.shape

### Numerical Variable Analysis:

#checking "Total Visits" vs Converted variable
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = leads)
plt.show()
### Inference
Median for converted and not converted leads are the close.
#checking "Page Views Per Visit" vs Converted variable

sns.boxplot(x=leads.Converted,y=leads['Page Views Per Visit'])
plt.show()
### Inference
Median for converted and unconverted leads is the same.
#checking "Total Time Spent on Website" vs Converted variable

sns.boxplot(x=leads.Converted, y=leads['Total Time Spent on Website'])
plt.show()
### Inference
Website should be made more engaging as Leads spending more time on the website are more likely to be converted, so to make leads spend more time.
## Correlation Matrix

cor = leads.corr()
cor
#Checking correlations of numeric values
## heatmap
plt.figure(figsize=(10,10))
sns.heatmap(cor, cmap="YlGnBu", annot=True)
plt.title("Correlation between Variables")
plt.show()
leads.shape
### Creation Dummy Variable

#list of categorical columns

categorical_cols= leads.select_dtypes(include=['object']).columns
categorical_cols
leads

#### So we have two columns 'A free copy of Mastering The Interview' & 'Do Not Email' to be changed to {0 ,1}

var_list =  ['A free copy of Mastering The Interview','Do Not Email']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the list
leads[var_list] = leads[var_list].apply(binary_map)
#getting dummies and dropping the first column and adding the results to the master dataframe
dummy = pd.get_dummies(leads[['Lead Origin','What is your current occupation',
                             'City']], drop_first=True)

leads = pd.concat([leads,dummy],1)
leads.shape

dummy = pd.get_dummies(leads['Specialization'], prefix  = 'Specialization')
dummy = dummy.drop(['Specialization_Not Specified'], 1)
leads = pd.concat([leads, dummy], axis = 1)
dummy = pd.get_dummies(leads['Lead Source'], prefix  = 'Lead Source')
dummy = dummy.drop(['Lead Source_Others'], 1)
leads = pd.concat([leads, dummy], axis = 1)

dummy = pd.get_dummies(leads['Tags'], prefix  = 'Tags')
dummy = dummy.drop(['Tags_Not Specified'], 1)
leads = pd.concat([leads, dummy], axis = 1)
dummy = pd.get_dummies(leads['Last Activity'], prefix  = 'Last Activity')
dummy = dummy.drop(['Last Activity_Others'], 1)
leads = pd.concat([leads, dummy], axis = 1)
dummy = pd.get_dummies(leads['Last Notable Activity'], prefix  = 'Last Notable Activity')
dummy = dummy.drop(['Last Notable Activity_Other_Notable_activity'], 1)
leads = pd.concat([leads, dummy], axis = 1)
#dropping original columns

leads.drop(categorical_cols,1,inplace = True)
leads.head()
## Logistic Regression Model Building:

### Spliting data to Train-Test:

from sklearn.model_selection import train_test_split

y = leads['Converted']
X=leads.drop('Converted', axis=1)
# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
### Scaling of Data:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols=X_train.select_dtypes(include=['float64', 'int64']).columns
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_train.head()
### Feature Selection Using RFE

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg, 18)    
rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
col
X_train.columns[~rfe.support_]
import statsmodels.api as sm
# Logistic regression model

X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()
col = col.drop('Lead Source_Welingak Website',1)
# Logistic regression model

X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
### Checking VIFs

# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
### VIF for all the variables looks good. So we don't need to drop any of these variables

# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]
# y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
### Creating a dataframe with the actual Converted flag and the predicted probabilities

y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()
### Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

y_train_pred_final['Predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()
from sklearn import metrics
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)
# Predicted       not_converted    Converted
# Actual
# not_converted        3741          161
# Converted            292          2116
# Let's check the overall accuracy
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))
### Metrics beyond simply accuracy

TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
# Calculate false postive rate - predicting Converted when lead does not have converted
print(FP/ float(TN+FP))
# positive predictive value 
print (TP / float(TP+FP))
# Negative predictive value
print (TN / float(TN+ FN))
### Plotting the ROC Curve

#### An ROC curve demonstrates several things:

1.It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).

2.The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.

3.The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)
### Finding Optimal Cutoff Point

Optimal cutoff probability is that prob where we get balanced sensitivity and specificity
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()
### From the curve above, 0.3 is the optimum point to take it as a cutoff probability.

y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2
TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))
# Positive predictive value 
print (TP / float(TP+FP))
# Negative predictive value
print (TN / float(TN+ FN))
### Observation:
### The model seems to be performing well. The ROC curve has a value of 0.97, which is very good.
### We have the following values for the Train Data:
### --> Accuracy : 92.14%
### --> Sensitivity : 91.49%
### --> Specificity : 92.54%
## Precision and Recall
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion
### Precision
TP / TP + FP
confusion[1,1]/(confusion[0,1]+confusion[1,1])
### Recall
TP / TP + FN
confusion[1,1]/(confusion[1,0]+confusion[1,1])
### Using sklearn utilities for the same
from sklearn.metrics import precision_score, recall_score
?precision_score
precision_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
recall_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
### Precision and recall tradeoff
from sklearn.metrics import precision_recall_curve
y_train_pred_final.Converted, y_train_pred_final.final_predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()
### Making predictions on the test set
num_cols=X_test.select_dtypes(include=['float64', 'int64']).columns

X_test[num_cols] = scaler.fit_transform(X_test[num_cols])

X_test.head()
X_test = X_test[col]
X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)
# Let's see the head
y_pred_1.head()
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Putting ProspectID to index
y_test_df['Prospect ID'] = y_test_df.index
# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})
# Rearranging the columns
y_pred_final = y_pred_final[['Prospect ID','Converted','Converted_prob']]
y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))
y_pred_final['final_Predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.3 else 0)

y_pred_final.head()
# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_Predicted)
confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_Predicted )
confusion2
TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
precision_score(y_pred_final.Converted , y_pred_final.final_Predicted)
recall_score(y_pred_final.Converted, y_pred_final.final_Predicted)
## Final Observation:
### Train Data:

Accuracy : 92.14%

Sensitivity : 91.49%

Specificity : 92.54%
### Test Data:
Accuracy : 92.57%

Sensitivity : 91.18%

Specificity : 93.46%
### The model seems to be performing well. Can recommend the company to making good calls based on this model.
## Thank You!!
