#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib

x_train_scaled=joblib.load( '../models/data/x_train_scaled.pkl')
x_test_scaled=joblib.load('../models/data/x_test_scaled.pkl')

y_train = joblib.load('../models/data/y_train.pkl')
y_test = joblib.load('../models/data/y_test.pkl')


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier()
}


# In[3]:


from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import clone
import pandas as pd



cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_macro',
    'recall': 'recall_macro',
    'f1': 'f1_macro'
}

results = []

for name, model in models.items():

    pipeline = Pipeline([
        ('model', clone(model))
    ])

    scores = cross_validate(
        pipeline,
        x_train_scaled,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results.append({
        'Model': name,
        'Accuracy': scores['test_accuracy'].mean(),
        'Precision': scores['test_precision'].mean(),
        'Recall': scores['test_recall'].mean(),
        'F1-score': scores['test_f1'].mean()
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by='F1-score',
    ascending=False
)

print(results_df.round(3))


# In[4]:


final_model = LogisticRegression(max_iter=1000)

final_model.fit(x_train_scaled, y_train)

y_pred = final_model.predict(x_test_scaled)


# In[5]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0,1,2]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['low','medium','hight']
)

disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()


# In[6]:


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("Accuracy :", accuracy_score(y_test, y_pred))

print("Precision :", precision_score(y_test, y_pred, average='macro'))

print("Recall :", recall_score(y_test, y_pred, average='macro'))

print("F1-score :", f1_score(y_test, y_pred, average='macro'))


# In[7]:


from sklearn.metrics import roc_auc_score

y_proba = final_model.predict_proba(x_test_scaled)

auc = roc_auc_score(
    y_test,
    y_proba,
    multi_class='ovr',
    average='macro'
)

print("ROC-AUC :", auc)


# In[8]:


models_params = {

    'Logistic Regression': {
        'model': LogisticRegression(max_iter=2000),
        'params': {
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__solver': ['lbfgs'],
            'model__class_weight': [None, 'balanced']
        }
    },

    'SVM': {
        'model': SVC(),
        'params': [
            {
                'model__kernel': ['linear'],
                'model__C': [0.1, 1, 10, 100]
            },
            {
                'model__kernel': ['rbf'],
                'model__C': [0.1, 1, 10, 100],
                'model__gamma': ['scale', 'auto']
            }
        ]
    },

    'XGBoost': {
        'model': XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=1
        ),
        'params': {
            'model__n_estimators': [100, 200, 300],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0]
        }
    }
}


# In[9]:


from sklearn.model_selection import GridSearchCV
results = []
best_models = {}

for name, config in models_params.items():

    print(f"\nOptimisation de : {name}")

    pipeline = Pipeline([
        ('model', config['model'])
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=config['params'],
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    grid.fit(x_train_scaled, y_train)

    best_models[name] = grid.best_estimator_

    results.append({
        'Model': name,
        'Best F1-score': grid.best_score_,
        'Best Parameters': grid.best_params_
    })

    print("Meilleur F1-score :", round(grid.best_score_, 4))
    print("Meilleurs paramètres :", grid.best_params_)


# In[12]:


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by='Best F1-score',
    ascending=False
)

print(results_df.to_string(index=False))


# In[14]:


best_model_name = results_df.iloc[0]['Model']

best_model = best_models[best_model_name]

print("Meilleur modèle :", best_model_name)


# In[15]:


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

y_pred = best_model.predict(x_test_scaled)

print(classification_report(
    y_test,
    y_pred,
    target_names=['Low', 'Medium', 'High']
))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Low', 'Medium', 'High']
)

disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.show()


# In[16]:


y_proba = best_model.predict_proba(x_test_scaled)

auc = roc_auc_score(
    y_test,
    y_proba,
    multi_class='ovr',
    average='macro'
)

print("ROC-AUC :", auc)


# In[17]:


import joblib

joblib.dump(
    best_model,
    '../models/model/model.pkl'
)

print("Modèle sauvegardé avec succès !") 

