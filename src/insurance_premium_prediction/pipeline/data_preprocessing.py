from src.insurance_premium_prediction.utils.common_utils import log
from src.insurance_premium_prediction.config.configuration import ConfigManager
from src.insurance_premium_prediction.components.data_preprocessing import DataPreprocessing


STAGE_NAME = "Data Preprocessing"

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigManager() # ConfigManager class
            data_preprocessing_config = config_manager.get_data_preprocessing_config() # get data_preprocessing_config

            process = DataPreprocessing(config=data_preprocessing_config) # data_preprocessing object
            data = process.remove_duplicate() # remove duplicate
            # data = process.mean_encoding(data=data) # apply mean encoding
            data = process.apply_encoding(data=data) # apply encoding
            # data = psrocess.handle_outliers(data=data) # handle outliers
            process.split_and_save_data(data=data) # split and save data

        except Exception as ex:
            raise ex


if __name__ == "__main__":
    try:
        config_manager = ConfigManager() # ConfigManager class
        log_file = config_manager.get_log_config().running_log # get the log file

        log(file_object=log_file, log_message=f"{str('>')*15} Stage: {STAGE_NAME} started {str('<')*15}")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        log(file_object=log_file, log_message=f"{str('>')*15} Stage: {STAGE_NAME} completed {str('<')*15} \n\n")

    except Exception as ex:
        raise ex 