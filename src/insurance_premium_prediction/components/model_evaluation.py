from src.insurance_premium_prediction.utils.common_utils import log, save_json_file, load_json_file
from src.insurance_premium_prediction.entity.config_entity import ModelEvaluationConfig
import pandas as pd
import numpy as np
import joblib
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error




class ModelEvaluation:
    """
        The `ModelEvaluation` class is responsible for calculating performance metrics for a machine learning model and saving the results in a JSON file. It also logs the model parameters, performance matrix, and the model itself into MLflow.

        Example Usage:
            config = ModelEvaluationConfig()
            eval_obj = ModelEvaluation(config)
            eval_obj.get_performance_metrics()
            eval_obj.log_into_mflow()

        Methods:
            __init__(self, config:ModelEvaluationConfig): Initializes the `ModelEvaluation` class with the provided configuration.
            get_performance_metrics(self): Calculates performance metrics for the machine learning model and saves the results in a JSON file.
            log_into_mflow(self): Logs the model parameters, performance matrix, and the model itself into MLflow.

        Fields:
            config: An instance of the `ModelEvaluationConfig` class that contains the necessary configuration parameters for model evaluation.
    """
    def __init__(self, config:ModelEvaluationConfig):
        self.config = config
    
    def get_performance_metrics(self):
        """
            This code defines a method called get_performance_metrics in the ModelEvaluation class. 
            The method calculates performance metrics for a machine learning model and saves the results in a JSON file.
        """
        try:
            log_file = self.config.log_file # mention log file
            train_data = pd.read_csv(self.config.train_data_path) # train data
            test_data = pd.read_csv(self.config.test_data_path) # test data

            X_train, Y_train = train_data.drop(self.config.y_feature, axis=1), train_data[self.config.y_feature]
            X_test, Y_test = test_data.drop(self.config.y_feature, axis=1), test_data[self.config.y_feature]

            scaler = StandardScaler() # standard scaler
            X_train_scale = scaler.fit_transform(X_train) # apply StandardScaler on X_train data
            X_test_scale = scaler.transform(X_test) # apply StandardScaler on X_test data

            model = joblib.load(self.config.model_path) # load the model

            y_train_predict = model.predict(X_train_scale) # predict based on train data
            y_test_predict = model.predict(X_test_scale) # predict based on test data

            performance_report = {
                "train_matrix": {
                    "r2_score": round(r2_score(y_true=Y_train, y_pred=y_train_predict), 2),
                    "mean_absolute_error": round(mean_absolute_error(y_true=Y_train, y_pred=y_train_predict), 2),
                    "mean_squared_error": round(mean_squared_error(y_true=Y_train, y_pred=y_train_predict), 2),
                    "root_mean_squared_error": round(np.sqrt(mean_squared_error(y_true=Y_train, y_pred=y_train_predict)), 2)
                },
                "test_matrix": {
                    "r2_score": round(r2_score(y_true=Y_test, y_pred=y_test_predict), 2),
                    "mean_absolute_error": round(mean_absolute_error(y_true=Y_test, y_pred=y_test_predict), 2),
                    "mean_squared_error": round(mean_squared_error(y_true=Y_test, y_pred=y_test_predict), 2),
                    "root_mean_squared_error": round(np.sqrt(mean_squared_error(y_true=Y_test, y_pred=y_test_predict)), 2)
                }
            }
            save_json_file(file_path=self.config.performance_report_path, report=performance_report) # save the performance matrix
            log(file_object=log_file, log_message=f"evaluate the performance of the model and save the performance matrix into {self.config.performance_report_path}") # logs

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be {ex}") # logs 
            raise ex


    def log_into_mflow(self):
        """
            Logs the model parameters, performance matrix, and the model itself into MLflow.

            Inputs:
            - self: The instance of the ModelEvaluation class.

            Example Usage:
            ```python
            config = ModelEvaluationConfig()
            eval_obj = ModelEvaluation(config)
            eval_obj.log_into_mflow()
            ```

            Code Analysis:
                1. Load the model and model parameters from the specified paths.
                2. Load the performance matrix from the specified path.
                3. Set the MLflow registry and tracking URIs.
                4. Start an MLflow run.
                5. Log the model parameters and performance metrics.
                6. If the tracking URL type is not "file", log the model with the registered model name. Otherwise, log the model without a registered model name.
                7. Log a success message.

            Outputs:
            - None. The method logs the model parameters, performance matrix, and model into MLflow.
        """
        try:
            log_file = self.config.log_file # mention log file
            model = joblib.load(self.config.model_path) # load the model
            model_params = load_json_file(self.config.model_params_path) # load the model params
            all_params = model_params["params"] # get the hyper parameters that is used to train the model
        
            performance_matrix = load_json_file(file_path=self.config.performance_report_path) # load the performance matrix
            matrices = performance_matrix["test_matrix"] # get the performance matrix

            mlflow.set_registry_uri(self.config.mlflow_uri)
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                mlflow.log_params(all_params) # logs the all parameters
                mlflow.log_metrics(matrices) # logs the metrics

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name=model_params["model_name"])
                else:
                    mlflow.sklearn.log_model(model, "model")

            log(file_object=log_file, log_message=f"successfully logs the model parameters, performance matrix and model into mlflow") # logs

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be {ex}") # logs 
            raise ex




if __name__ == "__main__":
    from src.insurance_premium_prediction.config.configuration import ConfigManager
    config_manager = ConfigManager()
    model_evaluation_config = config_manager.get_model_evaluation_config()

    eval_obj = ModelEvaluation(config=model_evaluation_config)
    eval_obj.get_performance_metrics()
    eval_obj.log_into_mflow()
