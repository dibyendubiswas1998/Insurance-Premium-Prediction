from pathlib import Path
from dataclasses import dataclass
import os



@dataclass(frozen=True)
class LogConfig:
    """
      Represents the configuration for logging in a Python application.

      Attributes:
          running_log (Path): The path to the running log file.
    """
    running_log: Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
        Represents the configuration for data ingestion.

        Attributes:
            s3_service_name (str): The name of the S3 service.
            s3_aws_access_key_id (str): The AWS access key ID for accessing the S3 service.
            s3_aws_secret_access_key (str): The AWS secret access key for accessing the S3 service.
            s3_region_name (str): The region name of the S3 service.
            s3_bucket_name (str): The name of the S3 bucket.
            s3_dataset (str): The name of the dataset within the S3 bucket.
            local_data_directory (Path): The local directory where the data will be stored.
            local_data_file_name (Path): The name of the data file.
            log_file (Path): The path to the log file.
    """
    s3_service_name: str
    s3_aws_access_key_id: str
    s3_aws_secret_access_key: str
    s3_region_name: str
    s3_bucket_name: str
    s3_dataset: str
    local_data_directory: Path
    local_data_file_name: Path
    log_file: Path

@dataclass(frozen=True)
class DataInfoConfig:
    """
        A data class that represents the configuration for data information.

        Attributes:
            columns (list): A list that represents the columns of the data.
            Y_feature_name (str): A string that represents the name of the target feature.
    """
    columns: list
    Y_feature_name: str


@dataclass(frozen=True)
class DataPreprocessingConfig:
    """
        Represents the configuration for data preprocessing.

        Attributes:
            data_path (Path): The path to the main data file.
            columns (list): A list of columns in the data that required.
            categorical_cols (list): A list of categorical columns in the data.
            numerical_cols (list): A list of numerical columns in the data.
            x_cols (list): A list of feature columns.
            y_feature (str): The target feature.
            train_data_path (Path): The path to the training data file.
            test_data_path (Path): The path to the test data file.
            encoded_metrics_file_path (Path): The path to the file where encoded metrics will be saved.
            test_size (float): The ratio of test data size to the total data size.
            random_state (int): The random state for data splitting.
            log_file (Path): The path to the log file.
    """
    data_path: Path
    columns: list
    categorical_cols: list
    numerical_cols: list
    x_cols: list
    y_feature: str
    train_data_path: Path
    test_data_path: Path
    encoded_metrics_file_path: Path
    test_size: float
    random_state: int
    log_file: Path


@dataclass(frozen=True)
class ModelTrainigConfig:
    """
        Represents the configuration for training a machine learning model.

        Attributes:
            x_cols (list): A list of feature column names.
            y_feature (str): The target feature column name.
            train_data_path (Path): The path to the training data file.
            test_data_path (Path): The path to the test data file.
            test_size (float): The proportion of the test data.
            random_state (int): The random seed for reproducibility.
            model_path (Path): The path to save the trained model.
            model_params_path (Path): The path to save the model parameters.
            rand_bootstrap (bool): A boolean indicating whether to use bootstrap samples for random forest.
            rand_max_depth (int): The maximum depth of the random forest.
            rand_max_features (int): The maximum number of features to consider for each split in random forest.
            rand_min_samples_leaf (int): The minimum number of samples required to be at a leaf node in random forest.
            rand_min_samples_split (int): The minimum number of samples required to split an internal node in random forest.
            rand_n_estimators (int): The number of trees in random forest.
            rand_random_state (int): The random seed for random forest.
            grad_n_estimators (int): The number of boosting stages to perform for gradient boosting.
            grad_learning_rate (float): The learning rate for gradient boosting.
            grad_max_depth (int): The maximum depth of the individual regression estimators in gradient boosting.
            grad_random_state (int): The random seed for gradient boosting.
            grad_subsample (int): The fraction of samples to be used for fitting the individual regression estimators in gradient boosting.
            log_file (Path): The path to save the log file.
    """
    x_cols: list
    y_feature: str
    train_data_path: Path
    test_data_path: Path
    test_size: float
    random_state: int
    model_path: Path
    model_params_path: Path
    # random forest regressor params:
    rand_bootstrap: bool
    rand_max_depth: int
    rand_max_features: int
    rand_min_samples_leaf: int
    rand_min_samples_split: int
    rand_n_estimators: int
    rand_random_state: int
    # gradient boosting regressor params:
    grad_n_estimators: int
    grad_learning_rate: float
    grad_max_depth: int
    grad_random_state: int
    grad_subsample: int
    log_file: Path