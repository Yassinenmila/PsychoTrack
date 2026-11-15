
from datetime import datetime

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator


def hello_ml():
    print("Hello PsychoTrack!")
    print("Machine Learning Pipeline started!")


with DAG(
    dag_id="psychotrack_ml_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    test_task = PythonOperator(
        task_id="test_ml",
        python_callable=hello_ml,
    )
