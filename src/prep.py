
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


def run_prep():

    print("Starting PsychoTrack preprocessing...")

    # 1. Définir les chemins
    BASE_DIR = Path(__file__).resolve().parent.parent

    DATA_PATH = BASE_DIR / "data" / "data-students.csv"
    OUTPUT_DIR = BASE_DIR / "models" / "data"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Charger les données
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded:", df.shape)

    # 3. Séparer les variables explicatives et la cible
    x = df.drop(columns=["Student_ID", "Mental_Health_Risk"])
    y = df["Mental_Health_Risk"]

    # 4. Séparation Train / Test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training set:", x_train.shape)
    print("Testing set:", x_test.shape)

    print("\nDistribution de y_train :")
    print(y_train.value_counts())

    print("\nPourcentage :")
    print((y_train.value_counts(normalize=True) * 100).round(2))

    # 5. Définir les colonnes
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

    # 6. Pipeline numérique
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 7. Pipeline catégoriel
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 8. Pipeline ordinal
    ord_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            categories=[
                ['Low', 'Moderate', 'High'],
                ['Poor', 'Average', 'Good']
            ]
        ))
    ])

    # 9. Pipeline binaire
    bin_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            drop='if_binary',
            handle_unknown='ignore'
        ))
    ])

    # 10. Combiner les pipelines
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols),
        ('ord', ord_pipeline, ord_cols),
        ('bin', bin_pipeline, bin_cols)
    ])

    # 11. Vérifier les valeurs ordinales
    for c in ord_cols:
        print(f"\n{c}")
        print(x_train[c].unique())

    # 12. Prétraitement des données
    print("\nFitting preprocessor...")

    x_train_scaled = preprocessor.fit_transform(x_train)

    x_test_scaled = preprocessor.transform(x_test)

    # 13. Encodage de la variable cible
    target_encoder = OrdinalEncoder(
        categories=[['Low', 'Medium', 'High']]
    )

    y_train = target_encoder.fit_transform(
        y_train.to_frame()
    ).ravel()

    y_test = target_encoder.transform(
        y_test.to_frame()
    ).ravel()

    # 14. Sauvegarder les données transformées
    joblib.dump(x_train_scaled, OUTPUT_DIR / "x_train_scaled.pkl")
    joblib.dump(x_test_scaled, OUTPUT_DIR / "x_test_scaled.pkl")

    joblib.dump(y_train, OUTPUT_DIR / "y_train.pkl")
    joblib.dump(y_test, OUTPUT_DIR / "y_test.pkl")

    # 15. Sauvegarder les objets de prétraitement
    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.pkl")
    joblib.dump(target_encoder, OUTPUT_DIR / "target_encoder.pkl")

    print("\nPreprocessing completed successfully!")

    # Retourner uniquement des informations simples pour Airflow
    return {
        "x_train_shape": x_train_scaled.shape,
        "x_test_shape": x_test_scaled.shape,
        "output_dir": str(OUTPUT_DIR)
    }
