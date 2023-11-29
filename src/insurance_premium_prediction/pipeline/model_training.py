from src.insurance_premium_prediction.utils.common_utils import log
from src.insurance_premium_prediction.config.configuration import ConfigManager
from src.insurance_premium_prediction.components.model_training import ModelTraining


STAGE_NAME = "Model Training"


class ModelTrainingPipeline:
    """
        Class responsible for training a model using the configuration provided by the ConfigManager class.

        Methods:
        - __init__(): Initializes the class.
        - main(): Orchestrates the model training process.
    """

    def __init__(self):
        pass

    def main(self):
        """
            Orchestrates the model training process.

            Raises:
            - Exception: If an error occurs during the model training process.
        """
        try:
            config_manager = ConfigManager() # ConfigManager class
            model_training_config = config_manager.get_model_training_config() # get model_training_config

            model_training_obj = ModelTraining(config=model_training_config) # model_training object
            model_training_obj.get_better_model() # get the best model by comparing the models          

        except Exception as ex:
            raise ex


if __name__ == "__main__":
    try:
        config_manager = ConfigManager() # ConfigManager class
        log_file = config_manager.get_log_config().running_log # get the log file

        log(file_object=log_file, log_message=f"{str('>')*15} Stage: {STAGE_NAME} started {str('<')*15}")
        obj = ModelTrainingPipeline()
        obj.main()
        log(file_object=log_file, log_message=f"{str('>')*15} Stage: {STAGE_NAME} completed {str('<')*15} \n\n")

    except Exception as ex:
        raise ex 