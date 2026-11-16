from datetime import datetime
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

from src.explore import run_eda
from src.prep import run_prep
from src.training import run_train


def hello_ml():
    print("Hello PsychoTrack!")
    print("Machine Learning Pipeline started!")


with DAG(
    dag_id="psychotrack_ml_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    description="Pipeline Machine Learning pour le projet PsychoTrack",
) as dag:

    test_task = PythonOperator(
        task_id="test_ml",
        python_callable=hello_ml,
    )

    eda_task = PythonOperator(
        task_id="exploratory_data_analysis",
        python_callable=run_eda,
    )

    prep_task = PythonOperator(
        task_id="data_prep",
        python_callable=run_prep,
    )

    train_task = PythonOperator(
        task_id = 'model_train',
        python_callable=run_train,
    )

    test_task >> eda_task >> prep_task >> train_task

