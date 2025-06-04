#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
#%%
#Read the excel file
data01 = pd.read_excel('student.xlsx')
#Extracted from a zip no Kaggle
#%%
#Removing not useful columns
data01.drop(columns=['Unnamed: 0'], inplace = True)
#%%
#Descriptive statistics for numerical variables
print(data01[['Student_Age', 'Weekly_Study_Hours']].describe())
#Categorical variables
#%%
#Renaming columns for an easier task
df = pd.DataFrame(data01)
df.rename(columns={
    'High_School_Type': 'Type',
    'Additional_Work': 'Work',
    'Sports_activity': 'Sports',
    'Listening_in_Class': 'Listening',
    'Project_work': 'Project'
}, inplace=True)
print(df)
#%%
#Checking data frame's columns
print(df.columns)
#Transforming categorical variables into dummies
df = pd.get_dummies(
    df,
    columns=['Sex', 'Work', 'Sports', 'Transportation', 'Reading', 'Notes', 'Listening', 'Project'],
    drop_first=False,
    dtype='int'
)

#%%
#Dropping unuseful columns
df.drop(columns=['Sex_Female', 'Work_No', 'Sports_No', 'Transportation_Bus', 'Reading_No', 'Notes_6', 'Notes_No', 'Listening_6', 'Listening_No', 'Project_No'], inplace=True)
#Renaming the remaining columns from the data wrangling
df.rename(columns={
    'Sex_Male': 'Sex',
    'Work_Yes': 'Work',
    'Sports_Yes': 'Sports',
    'Transportation_Private': 'Transportation',
    'Reading_Yes': 'Reading',
    'Notes_Yes': 'Notes',
    'Listening_Yes': 'Listening',
    'Project_Yes': 'Project'
    }, inplace=True)
print(df)
#Sex 1 = male, Work 1 = Yes, Sprots 1 = Yes, Transportation 1 = public, Reading 1 - yes, Listening 1 = yes, Project 1 = yes

#%%
df.info()
X = df

#%%
# Converting grades in alphabetical letters to numbers
# Substitution dictionary
#grades_to_numbers_series = df['Grade']


#%% 
df_temp = df
y = df_temp

grades_to_numbers = {
    'AA': '4',
    'BA': '3.5',
    'BB': '3',
    'CB': '2.5',
    'CC': '2',
    'DC': '1.5',
    'DD': '1.0',
    'Fail': '0'
}
#y = y.astype(float)
#print(y_train['Grade'].unique())

# Convertendo as notas para números
y['Grade'] = df_temp['Grade'].replace(grades_to_numbers)
#y_train['Grade_str'] = y_train['Grade'].replace(grades_to_numbers)

df_temp['Numerical_grade'] = y['Grade'].astype(float)
#y_train['Numerical_grade'] = df['Grade_str'].astype(float)


#%%
# Basic information about the dataframe
df_temp.info()
df.info()
df.drop(columns=['Id', 'Student_Age', 'Weekly_Study_Hours', 'Sex', 'Work', 'Sports', 'Transportation', 'Reading', 'Notes', 'Listening', 'Project'])
#%%
#Making clear which are X and which are Y
df_temp.info()
X = df_temp.drop(columns=['Id'])
y = df['Grade'].astype(float)

#%%
#X_train.columns
#X_train.info()
#X_train.drop(columns=['Type', 'Scholarship', 'Attendance'],
#             inplace=True);
X.columns
X.info()
X = X.drop(columns=['Type', 'Scholarship', 'Attendance', 'Numerical_grade'],
             inplace=True);

df_temp.info()

#y_train.drop(columns='Grade',
#             inplace=True)
#y_test.drop(columns='Grade',
#            inplace=True)
y_train = df['Numerical_grade']
y_test = df['Numerical_grade']
print(X.shape)
print(y.shape)
print(X.tail)
print(y.tail)
#%%\
#Tearing apart train and tests samples
#Let's just choose 70% of the sample for tests and 30% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#%%
#DECISION TREE
tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X_train, y_train)
#%%
#Plotting the tree
plt.figure(figsize=(20,10), dpi=100)
plot_tree(tree_reg,
          max_depth=3,
          feature_names=X.columns.tolist(),
          filled=True,
          node_ids=True,
          precision=3)
plt.show()
#%%
