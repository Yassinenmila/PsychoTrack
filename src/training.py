
from pathlib import Path

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)
from sklearn.base import clone

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def run_train():

    print("Starting PsychoTrack model training...")

    # 1. Définir les chemins
    BASE_DIR = Path(__file__).resolve().parent.parent

    DATA_DIR = BASE_DIR / "models" / "data"
    MODEL_DIR = BASE_DIR / "models" / "model"
    REPORT_DIR = BASE_DIR / "reports"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Charger les données prétraitées
    x_train_scaled = joblib.load(DATA_DIR / "x_train_scaled.pkl")
    x_test_scaled = joblib.load(DATA_DIR / "x_test_scaled.pkl")

    y_train = joblib.load(DATA_DIR / "y_train.pkl")
    y_test = joblib.load(DATA_DIR / "y_test.pkl")

    print("Training data:", x_train_scaled.shape)
    print("Testing data:", x_test_scaled.shape)

    # 3. Définir les modèles
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "XGBoost": XGBClassifier()
    }

    # 4. Définir la validation croisée
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro"
    }

    # 5. Comparer les modèles
    results = []

    for name, model in models.items():

        print(f"\nEvaluating: {name}")

        pipeline = Pipeline([
            ("model", clone(model))
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
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1-score": scores["test_f1"].mean()
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1-score",
        ascending=False
    )

    print("\nModel comparison:")
    print(results_df.round(3))

    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    # 6. Évaluation initiale de Logistic Regression
    print("\nInitial Logistic Regression evaluation")

    final_model = LogisticRegression(max_iter=1000)

    final_model.fit(x_train_scaled, y_train)

    y_pred = final_model.predict(x_test_scaled)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="macro"))
    print("Recall:", recall_score(y_test, y_pred, average="macro"))
    print("F1-score:", f1_score(y_test, y_pred, average="macro"))

    y_proba = final_model.predict_proba(x_test_scaled)

    auc = roc_auc_score(
        y_test,
        y_proba,
        multi_class="ovr",
        average="macro"
    )

    print("ROC-AUC:", auc)

    # 7. Définir les modèles et hyperparamètres
    models_params = {

        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000),
            "params": {
                "model__C": [0.01, 0.1, 1, 10, 100],
                "model__solver": ["lbfgs"],
                "model__class_weight": [None, "balanced"]
            }
        },

        "SVM": {
            "model": SVC(),
            "params": [
                {
                    "model__kernel": ["linear"],
                    "model__C": [0.1, 1, 10, 100]
                },
                {
                    "model__kernel": ["rbf"],
                    "model__C": [0.1, 1, 10, 100],
                    "model__gamma": ["scale", "auto"]
                }
            ]
        },

        "XGBoost": {
            "model": XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=1
            ),
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.1, 0.2],
                "model__max_depth": [3, 5, 7],
                "model__subsample": [0.8, 1.0]
            }
        }
    }

    # 8. Optimisation avec GridSearchCV
    print("\nStarting hyperparameter optimization...")

    results = []
    best_models = {}

    for name, config in models_params.items():

        print(f"\nOptimisation de : {name}")

        pipeline = Pipeline([
            ("model", config["model"])
        ])

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=config["params"],
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
            refit=True
        )

        grid.fit(x_train_scaled, y_train)

        best_models[name] = grid.best_estimator_

        results.append({
            "Model": name,
            "Best F1-score": grid.best_score_,
            "Best Parameters": grid.best_params_
        })

        print("Meilleur F1-score:", round(grid.best_score_, 4))
        print("Meilleurs paramètres:", grid.best_params_)

    # 9. Comparer les meilleurs modèles
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Best F1-score",
        ascending=False
    )

    print("\nHyperparameter optimization results:")
    print(results_df.to_string(index=False))

    results_df.to_csv(
        REPORT_DIR / "hyperparameter_results.csv",
        index=False
    )

    # 10. Sélectionner le meilleur modèle
    best_model_name = results_df.iloc[0]["Model"]

    best_model = best_models[best_model_name]

    print("\nBest model:", best_model_name)

    # 11. Évaluation finale
    y_pred = best_model.predict(x_test_scaled)

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Low", "Medium", "High"]
    ))

    # 12. Matrice de confusion
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1, 2]
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low", "Medium", "High"]
    )

    disp.plot(cmap="Blues")

    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.savefig(REPORT_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # 13. ROC-AUC
    y_proba = best_model.predict_proba(x_test_scaled)

    auc = roc_auc_score(
        y_test,
        y_proba,
        multi_class="ovr",
        average="macro"
    )

    print("\nFinal ROC-AUC:", auc)

    # 14. Sauvegarder le meilleur modèle
    joblib.dump(
        best_model,
        MODEL_DIR / "model.pkl"
    )

    print("\nBest model saved successfully!")
    print("Training completed!")

    # 15. Retourner des informations simples pour Airflow
    return {
        "best_model": best_model_name,
        "best_f1_score": float(results_df.iloc[0]["Best F1-score"]),
        "roc_auc": float(auc),
        "model_path": str(MODEL_DIR / "model.pkl")
    }
