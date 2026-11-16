
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Chemin racine du projet PsychoTrack
BASE_DIR = Path(__file__).resolve().parent.parent

# Chemin du dataset
DATA_PATH = BASE_DIR / "data" / "data-students.csv"

def run_eda():

    df = pd.read_csv(DATA_PATH)

    print(df.head())
    print("Colonnes :", df.columns)
    print("Dimensions :", df.shape)
    df.info()

    df = df.drop(columns='Student_ID')

    print("\nValeurs manquantes :")
    print(df.isna().sum())

    categorical_cols = df.select_dtypes(include='object').columns

    print("\nNombre de catégories :")
    print(df.select_dtypes(include='object').nunique())

    for col in categorical_cols:
        print(f"\n{col}")
        print(df[col].value_counts(dropna=False))
        print(df[col].unique())

    print("\nNombre de doublons :")
    print(df.duplicated().sum())

    print("\nStatistiques descriptives :")
    print(df.describe().T)

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
            print(outliers[col])

    target = 'Mental_Health_Risk'

    print("\nDistribution de la cible :")
    print(df[target].value_counts())

    print("\nPourcentage des classes :")
    print(df[target].value_counts(normalize=True) * 100)

    plt.figure(figsize=(8, 5))

    sns.countplot(data=df, x=target)

    plt.title("Distribution de la variable cible")
    plt.xlabel("Classes")
    plt.ylabel("Nombre d'observations")

    plt.show()

    print("\nMoyennes selon le risque :")
    print(df.groupby(target).mean(numeric_only=True).T)

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

    corr = df.select_dtypes(include='number').corr()

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        fmt='.2f'
    )

    plt.title("Matrice de corrélation")

    plt.show()

    for col in numeric_cols:

        plt.figure(figsize=(8, 5))

        sns.boxplot(data=df, x=target, y=col)

        plt.title(f'{col} selon le niveau de risque')
        plt.xlabel('Niveau de risque')
        plt.ylabel(col)

        plt.show()

    print("\nAnalyse exploratoire terminée !")

    return df

# run_eda()
import os 
print(os.listdir())