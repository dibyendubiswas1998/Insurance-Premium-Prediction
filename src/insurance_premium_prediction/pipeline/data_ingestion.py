from src.insurance_premium_prediction.config.configuration import ConfigManager
from src.insurance_premium_prediction.components.data_ingestion import DataIngestion
from src.insurance_premium_prediction.utils.common_utils import log 


STAGE_NAME = "Data Ingestion"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigManager() # ConfigManager class
            data_ingestion_config = config_manager.get_data_ingestion_config() # get data_ingestion_config

            dd = DataIngestion(config=data_ingestion_config) # data_ingestion object
            dd.load_and_save_data() # load and save the data process

        except Exception as ex:
            raise ex




if __name__ == "__main__":
    try:
        config_manager = ConfigManager() # ConfigManager class
        log_file = config_manager.get_log_config().running_log # get the log file

        log(file_object=log_file, log_message=f"{str('>')*15} Stage: {STAGE_NAME} started {str('<')*15}")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        log(file_object=log_file, log_message=f"{str('>')*15} Stage: {STAGE_NAME} completed {str('<')*15} \n\n")

    except Exception as ex:
        raise ex 