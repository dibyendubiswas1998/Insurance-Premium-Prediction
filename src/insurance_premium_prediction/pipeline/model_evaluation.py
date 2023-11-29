from src.insurance_premium_prediction.utils.common_utils import log
from src.insurance_premium_prediction.config.configuration import ConfigManager
from src.insurance_premium_prediction.components.model_evaluation import ModelEvaluation


STAGE_NAME = "Model Evaluation"


class ModelEvaluationTrainingPipeline:
    """
        A class responsible for evaluating a machine learning model's performance and logging the results into a 
        workflow management system.
    """
    def __init__(self):
        """
            Initializes the ModelEvaluationPipeline class.
        """
        pass

    def main(self):
        """
            Executes the main functionality of the class, which includes retrieving the model evaluation configuration, 
            calculating performance metrics, and logging the results.
        """
        try:
            config_manager = ConfigManager()
            model_evaluation_config = config_manager.get_model_evaluation_config()

            eval_obj = ModelEvaluation(config=model_evaluation_config)
            eval_obj.get_performance_metrics()
            eval_obj.log_into_mflow()

        except Exception as ex:
            raise ex
        
        

if __name__ == "__main__":
    try:
        config_manager = ConfigManager() # ConfigManager class
        log_file = config_manager.get_log_config().running_log # get the log file

        log(file_object=log_file, log_message=f"{str('>')*15} Stage: {STAGE_NAME} started {str('<')*15}")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        log(file_object=log_file, log_message=f"{str('>')*15} Stage: {STAGE_NAME} completed {str('<')*15} \n\n")

    except Exception as ex:
        raise ex 