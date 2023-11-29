import os
from src.insurance_premium_prediction.constants import *
from src.insurance_premium_prediction.entity.config_entity import *
from src.insurance_premium_prediction.utils.common_utils import read_params



class ConfigManager:
    def __init__(self, secrect_file_path=SECRECT_FILE_PATH, config_file_path=CONFIG_FILE_PATH, 
                 params_file_path=PARAMS_FILE_PATH):
        
        self.secrect = read_params(secrect_file_path) # read information from config/secrect.yaml file
        self.config = read_params(config_file_path) # read information from config/config.yaml file
        self.params = read_params(params_file_path) # read information from params.yaml file
    

    def get_log_config(self) -> LogConfig:
        """
            Retrieves the log configuration from the config.yaml file.

            Returns:
                LogConfig: The log configuration as a LogConfig object.

            Raises:
                Exception: If there is an error retrieving the log configuration.
        """
        try:
            log_file_config = LogConfig(running_log=self.config.logs.log_file)
            return log_file_config

        except Exception as ex:
            raise ex
        

    def get_data_info_config(self) -> DataInfoConfig:
        """
            Retrieves the data information configuration from the secrect.yaml file and returns it as a DataInfoConfig object.

            Returns:
                DataInfoConfig: The data information configuration object.

            Raises:
                Exception: If there is an error retrieving the data information configuration.
        """
        try:
            data_info_config = DataInfoConfig(
                columns=self.secrect.data_info.columns,
                Y_feature_name=self.secrect.data_info.Y_feature
            )
            return data_info_config

        except Exception as ex:
            raise ex
    

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
            Retrieves the data ingestion configuration by creating a `DataIngestionConfig` object and populating it with values from the `secrect.yaml` and `config.yaml` files.

            Returns:
                DataIngestionConfig: The data ingestion configuration object containing the following properties:
                    - s3_service_name: The name of the S3 service.
                    - s3_aws_access_key_id: The AWS access key ID for S3.
                    - s3_aws_secret_access_key: The AWS secret access key for S3.
                    - s3_region_name: The region name for S3.
                    - s3_bucket_name: The name of the S3 bucket.
                    - s3_dataset: The name of the dataset in the S3 bucket.
                    - local_data_directory: The local directory where the data will be stored.
                    - local_data_file_name: The name of the raw data file.
                    - log_file: The file path for the log file.
        """
        try:
            data_ingestion_config = DataIngestionConfig(
                s3_service_name=self.secrect.s3_bucket_access.service_name,
                s3_aws_access_key_id=self.secrect.s3_bucket_access.acess_key,
                s3_aws_secret_access_key=self.secrect.s3_bucket_access.secret_key,
                s3_region_name=self.secrect.s3_bucket_access.region_name,
                s3_bucket_name=self.secrect.s3_bucket_access.bucket_name,
                s3_dataset=self.secrect.s3_bucket_access.dataset_name,
                local_data_directory=self.config.artifacts.data.data_dir,
                local_data_file_name=self.config.artifacts.data.raw_data_file_name,
                log_file=self.config.logs.log_file
            )
            return data_ingestion_config

        except Exception as ex:
            raise ex
        

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        """
            Retrieves the data preprocessing configuration.

            Returns:
                DataPreprocessingConfig: The data preprocessing configuration.

            Example Usage:
                # Initialize the ConfigManager object
                config_manager = ConfigManager()

                # Call the get_data_preprocessing_config method
                data_preprocessing_config = config_manager.get_data_preprocessing_config()

                # Print the data preprocessing configuration
                print(data_preprocessing_config)

            Expected output:
                DataPreprocessingConfig(
                data_path='path/to/raw_data_file',
                columns=['col1', 'col2', ...],
                categorical_cols=['col1', 'col2', ...],
                numerical_cols=['col3', 'col4', ...],
                x_cols=['col1', 'col2', ...],
                y_feature='target',
                train_data_path='path/to/train_data',
                test_data_path='path/to/test_data',
                encoded_metrics_file_path='path/to/encoded_metrics_file',
                test_size=0.2,
                random_state=42,
                log_file='path/to/log_file'
                )
        """
        try:
            data_preprocessing_config = DataPreprocessingConfig(
                data_path=self.config.artifacts.data.raw_data_file_name,
                columns=self.secrect.data_info.columns,
                categorical_cols=self.secrect.data_info.categorical_cols,
                numerical_cols=self.secrect.data_info.numerical_cols,
                x_cols=self.secrect.data_info.x_cols,
                y_feature=self.secrect.data_info.Y_feature,
                train_data_path=self.config.artifacts.processed_data.train_path,
                test_data_path=self.config.artifacts.processed_data.test_path,
                encoded_metrics_file_path=self.config.artifacts.processed_data.encoded_metrics_file_path,
                test_size=self.config.split_ratio.test_size,
                random_state=self.config.split_ratio.random_state,
                log_file=self.config.logs.log_file
            )
            return data_preprocessing_config

        except Exception as ex:
            raise ex


if __name__ == "__main__":
    cc = ConfigManager()
    print(cc.get_data_preprocessing_config())
