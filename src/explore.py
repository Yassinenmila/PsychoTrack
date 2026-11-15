#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd

df = pd.read_csv('../data/data-students.csv')

df


# In[18]:


df.columns


# In[19]:


df.head()


# In[20]:


df.shape


# In[22]:


df.isna().sum()


# In[23]:


# from sklearn.impute import SimpleImputer

# numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

# numeric_imputer = SimpleImputer(strategy="median")

# df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])


# In[24]:


df.isna().sum()


# In[25]:


df["Diet_Quality"].value_counts(dropna=False)


# In[26]:


df.isna().sum()


# In[27]:


df.duplicated().sum()


# In[28]:


df.describe().T


# In[29]:


numeric_cols = df.select_dtypes(include="number").columns

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    if len(outliers) > 0:
        print(f"\nColonne : {col}")
        print(outliers[["Student_ID", col]])


# In[30]:


df['Mental_Health_Risk'].value_counts()


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x='Mental_Health_Risk')

plt.title("Distribution de la variable cible")
plt.xlabel("Classes")
plt.ylabel("Nombre d'observations")

plt.show()


# In[33]:


target = 'Mental_Health_Risk'

df.groupby(target).mean(numeric_only=True).T


# In[34]:


plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x=target,
    y='Sleep_Hours'
)

plt.title('Relation entre les heures de sommeil et le risque')
plt.xlabel('Niveau de risque')
plt.ylabel('Heures de sommeil')

plt.show()

