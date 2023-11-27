import os
import boto3
import pandas as pd
from src.insurance_premium_prediction.utils.common_utils import log, clean_prev_dirs_if_exis, create_dir
from src.insurance_premium_prediction.entity.config_entity import DataIngestionConfig



class DataIngestion:
    """
        The DataIngestion class is responsible for loading data from two datasets stored in an S3 bucket, concatenating them
        together, and saving the resulting dataframe to a local directory. It also logs various messages throughout the process.
    """
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config
    

    def load_and_save_data(self):
        try:
            log_file = self.config.log_file # mention log file

            s3_client = boto3.resource(
                service_name = self.config.s3_service_name,
                region_name = self.config.s3_region_name,
                aws_access_key_id = self.config.s3_aws_access_key_id,
                aws_secret_access_key = self.config.s3_aws_secret_access_key
            )
            log(file_object=log_file, log_message=f"configure the s3 details") # logs

            object = s3_client.Bucket(self.config.s3_bucket_name).Object(self.config.s3_dataset).get()  # load the object_1
            df = pd.read_csv(object['Body'])   # load the dataset_1
            log(file_object=log_file, log_message=f"download the dataset from s3 bucket, {self.config.s3_dataset}") # logs 

            clean_prev_dirs_if_exis(dir_path=self.config.local_data_directory) # remove the directory if it exists
            create_dir(dirs=[self.config.local_data_directory]) # create the fresh directory
            log(file_object=log_file, log_message=f"clean and then create the directory {self.config.local_data_directory}") # logs 

            df.to_csv(self.config.local_data_file_name, index=None) # save the data
            log(file_object=log_file, log_message=f"save the data to local directory, {self.config.local_data_directory}") # logs 
        

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"cerror will be {ex}") # logs 
            raise ex
        


if __name__ == "__main__":
    from src.insurance_premium_prediction.config.configuration import ConfigManager
    config_manager = ConfigManager()
    data_ingestion_config = config_manager.get_data_ingestion_config()

    dd = DataIngestion(config=data_ingestion_config)
    dd.load_and_save_data()
