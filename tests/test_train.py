"""
    Tests in training pipeline
"""

import pytest
import pandas as pd
import numpy as np

from src.train.train_baseline import train_baseline
from src.train.tune import tune_model 


@pytest.fixture
def mock_pro_path(tmp_path):
    pro_path = tmp_path / "data" / "processed"
    pro_path.mkdir(parents=True, exist_ok=True)
    return pro_path

@pytest.fixture
def mock_model_path(tmp_path):
    model_path = tmp_path / "models"
    model_path.mkdir()
    return model_path

@pytest.fixture
def mock_mlflow_path(tmp_path):
    mlflow_path = tmp_path / "mlruns"
    mlflow_path.mkdir()
    return mlflow_path


def mock_data():
    trans_train_df = pd.DataFrame({
        "cat": [1,2,3,4,5],
        "cont": [1,2,3,4,5],
        "loss": [100,200,300,400,500],
        "log_loss": [np.log(100*(i+1)) for i in range(5)]
    })
    trans_valid_df = pd.DataFrame({
        "cat": [6,7,8,9,10],
        "cont": [6,7,8,9,10],
        "loss": [600,700,800,900,1000],
        "log_loss": [np.log(100*i) for i in range(6,11)]
    })
    return trans_train_df, trans_valid_df
    


# train_baseline.py
def test_train_baseline(mock_pro_path, mock_model_path) -> None:
    trans_train_df, trans_valid_df = mock_data()
    trans_train_df.to_parquet(mock_pro_path / "transformed_train.parquet")
    trans_valid_df.to_parquet(mock_pro_path / "transformed_valid.parquet")
    
    model, mae = train_baseline(pro_path=mock_pro_path, model_path=mock_model_path)

    assert (mock_model_path / "baseline" / "baseline_lgb.txt").exists()
    print("Train baseline model test passed")


# tune.py
def test_tune(mock_pro_path, mock_model_path, mock_mlflow_path) -> None:
    trans_train_df, trans_valid_df = mock_data()
    trans_train_df.to_parquet(mock_pro_path / "transformed_train.parquet")
    trans_valid_df.to_parquet(mock_pro_path / "transformed_valid.parquet")

    best_params, best_iteration, best_mae = tune_model(
        pro_path=mock_pro_path,
        model_path=mock_model_path,
        mlflow_path=mock_mlflow_path,
        n_trials=2
    )

    assert (mock_model_path / "retrain_lgb_config.yaml").exists() 
    assert isinstance(best_params, dict) and best_params
    print("Model tuning test passed")


 
