#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

df = pd.read_csv('../data/data-students.csv')
df


# In[3]:


x= df.drop(columns=['Student_ID','Mental_Health_Risk'])
y=df['Mental_Health_Risk']


# In[4]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[5]:


print("Distribution de y_train :")
print(y_train.value_counts())

print("\nPourcentage :")
print((y_train.value_counts(normalize=True) * 100).round(2))


# In[6]:


num_cols = [
    'Age',
    'Year_of_Study',
    'Sleep_Hours',
    'Study_Hours_Per_Week',
    'GPA',
    'Financial_Stress',
    'Social_Media_Hours',
    'Screen_Time_Hours',
    'Family_Support',
    'Attendance_Rate'
]

cat_cols = [
    'Gender'
]

ord_cols = [
    'Physical_Activity',
    'Diet_Quality'
]

bin_cols = [
    'Part_Time_Job',
    'Previous_Mental_Health_History'
]


# In[7]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder ,LabelEncoder , OrdinalEncoder
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

ord_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(
        categories=[
            ['Low', 'Moderate', 'High'],
            ['Poor', 'Average', 'Good']
        ]
    ))
])

bin_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(
        drop='if_binary',
        handle_unknown='ignore'
    ))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols),
    ('ord', ord_pipeline, ord_cols),
    ('bin', bin_pipeline, bin_cols)
])


# In[8]:


for c in ord_cols:
    print(f"\n{c}")
    print(x_train[c].unique())
    print(x_train[c].dtype)


# In[9]:


x_train_scaled = preprocessor.fit_transform(x_train)
x_test_scaled=preprocessor.transform(x_test)




# In[17]:


from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(
    categories=[['Low', 'Medium', 'High']]
)

y_train = encoder.fit_transform(y_train.to_frame()).ravel()
y_test = encoder.transform(y_test.to_frame()).ravel()




# In[18]:


import joblib

joblib.dump(x_train_scaled, '../models/data/x_train_scaled.pkl')
joblib.dump(x_test_scaled, '../models/data/x_test_scaled.pkl')

joblib.dump(y_train, '../models/data/y_train.pkl')
joblib.dump(y_test, '../models/data/y_test.pkl')

