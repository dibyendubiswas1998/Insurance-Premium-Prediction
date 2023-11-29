from src.insurance_premium_prediction.utils.common_utils import log, save_model, save_json_file
from src.insurance_premium_prediction.entity.config_entity import ModelTrainigConfig
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score



class ModelTraining:
    """
        The `ModelTraining` class is responsible for training and evaluating two regression models (Random Forest and Gradient Boosting) for insurance premium prediction. It loads and splits the train and test data, applies feature scaling using StandardScaler, trains the models with the best possible parameters, compares their performance based on the R2 score, and saves the model with the higher score.

        Example Usage:
            # Create an instance of ModelTraining class
            model_obj = ModelTraining(config=model_training_config)

            # Train and evaluate the models
            model_obj.get_better_model()

        Main functionalities:
        - Load and split the train and test data for insurance premium prediction.
        - Apply feature scaling using StandardScaler.
        - Train a Random Forest regressor model with the best possible parameters.
        - Train a Gradient Boosting regressor model with the best possible parameters.
        - Compare the performance of the two models based on the R2 score.
        - Save the model with the higher R2 score.

        Methods:
        - load_and_split_data(): Load and split the train and test data, and apply feature scaling using StandardScaler.
        - train_random_forest_model(): Train a Random Forest regressor model with the best possible parameters.
        - train_gradient_boost_model(): Train a Gradient Boosting regressor model with the best possible parameters.
        - get_better_model(): Train and evaluate the Random Forest and Gradient Boosting models, compare their performance, and save the model with the higher R2 score.

        Fields:
        - config: An instance of the ModelTrainigConfig class that stores the configuration parameters for model training.
    """
    def __init__(self, config:ModelTrainigConfig):
        self.config = config
    
    def load_and_split_data(self):
        """
            Load and split the train and test data for the insurance premium prediction model.
            Apply StandardScaler to scale the input features.

            Returns:
            X_train_scale (array-like): Scaled input features of the train data.
            Y_train (array-like): Target variable of the train data.
            X_test_scale (array-like): Scaled input features of the test data.
            Y_test (array-like): Target variable of the test data.
        """
        try:
            log_file = self.config.log_file # mention log file
            # train data
            train_data = pd.read_csv(self.config.train_data_path) # load the train data       
            X_train, Y_train = train_data.drop(self.config.y_feature, axis=1), train_data[self.config.y_feature]

            # test data:
            test_data = pd.read_csv(self.config.test_data_path) # load the test data  
            X_test, Y_test = test_data.drop(self.config.y_feature, axis=1), test_data[self.config.y_feature]
        
            scaler = StandardScaler() # StandardScaler
            X_train_scale = scaler.fit_transform(X_train) # apply StandardScaler on X_train data
            X_test_scale = scaler.transform(X_test) # apply StandardScaler on X_test data

            log(file_object=log_file, log_message=f"get train and test data") # logs
            return X_train_scale, Y_train, X_test_scale, Y_test # return train and test datasets

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be {ex}") # logs 
            raise ex
        

    def train_random_forest_model(self):
        """
            Trains a random forest regressor model using the best possible parameters.

            Returns:
            - random_forest: Trained random forest regressor model.
            - params: Best parameters used for training the model.
        """
        try:
            log_file = self.config.log_file # mention log file
            X_train_scale, Y_train, X_test_scale, Y_test = self.load_and_split_data() # get the train and test data
            params = {"n_estimators": self.config.rand_n_estimators, 
                      "bootstrap": self.config.rand_bootstrap,
                      "max_depth": self.config.rand_max_depth, 
                      "max_features": self.config.rand_max_features,
                      "min_samples_leaf": self.config.rand_min_samples_leaf, 
                      "min_samples_split": self.config.rand_min_samples_split,
                      "random_state": self.config.rand_random_state
                    }
        
            random_forest = RandomForestRegressor(n_estimators=self.config.rand_n_estimators,
                                                  bootstrap=self.config.rand_bootstrap,
                                                  max_depth=self.config.rand_max_depth,
                                                  max_features=self.config.rand_max_features,
                                                  min_samples_leaf=self.config.rand_min_samples_leaf,
                                                  min_samples_split=self.config.rand_min_samples_split,
                                                  random_state=self.config.rand_random_state
                                                  ) # create model with best possible parameters
            random_forest.fit(X_train_scale, Y_train) # train the model
        
    
            log(file_object=log_file, log_message=f"train the  random forest regressor model using best parameters: {params}") # logs
            return random_forest, params # return the random forest model and best parameters

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be {ex}") # logs 
            raise ex
    

    def train_gradient_boost_model(self):
        """
            Trains a gradient boosting regressor model using the best possible parameters.

            Returns:
            - gradient_boosting: Trained gradient boosting regressor model.
            - params: Best parameters used for training the model.
        """
        try:
            log_file = self.config.log_file # mention log file
            X_train_scale, Y_train, X_test_scale, Y_test = self.load_and_split_data() # get the train and test data
            params = {"n_estimators": self.config.grad_n_estimators,
                      "learning_rate": self.config.grad_learning_rate,
                      "max_depth": self.config.grad_max_depth,
                      "random_state": self.config.grad_random_state,
                      "subsample": self.config.grad_subsample
                    }

            gradient_boosting = GradientBoostingRegressor(n_estimators=self.config.grad_n_estimators,
                                                          learning_rate=self.config.grad_learning_rate,
                                                          max_depth=self.config.grad_max_depth,
                                                          random_state=self.config.grad_random_state,
                                                          subsample=self.config.grad_subsample) # create model with best possible parameters
            gradient_boosting.fit(X_train_scale, Y_train) # train the model

            log(file_object=log_file, log_message=f"train the gradient boosting regressor model using best parameters: {params}") # logs
            return gradient_boosting, params

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be {ex}") # logs 
            raise ex


    def get_better_model(self):
        """
            Trains two regression models (Random Forest and Gradient Boosting) using the best possible parameters. 
            Compares the performance of the two models based on the R2 score and saves the model with the higher score.

            Inputs:
            - self: The instance of the ModelTraining class.
            - X_train_scale: Scaled input features of the train data.
            - Y_train: Target variable of the train data.
            - X_test_scale: Scaled input features of the test data.
            - Y_test: Target variable of the test data.

            Outputs:
            - The trained model with the higher R2 score.
            - The parameters used for training the model.
        """
        try:
            log_file = self.config.log_file # mention log file
            X_train_scale, Y_train, X_test_scale, Y_test = self.load_and_split_data() # get the train and test data

            # load the models:
            random_forest, random_forest_params = self.train_random_forest_model() # random forest regressor
            gradient_boosting, gradient_boosting_params = self.train_gradient_boost_model() # gradient boost regressor
        
            # predict based on random forest regressor:
            random_forest_predict = random_forest.predict(X_test_scale) # get the prediction
            random_forest_score = r2_score(y_true=Y_test, y_pred=random_forest_predict) # get the r2 score

            # predict based on gradient boosting regressor:
            gradient_boosting_predict = gradient_boosting.predict(X_test_scale) # get the prediction
            gradient_boosting_score = r2_score(y_true=Y_test, y_pred=gradient_boosting_predict) # get the r2 score

            # compare the two models based on r2 score:
            if gradient_boosting_score > random_forest_score:
                gradient_boosting_params_ = {
                    "model_name": 'GradientBoostingRegressor',
                    "params": gradient_boosting_params,
                    "score": gradient_boosting_score
                }
                save_model(model_name=gradient_boosting, model_path=self.config.model_path) # save the gradient_boosting model
                save_json_file(file_path=self.config.model_params_path, report=gradient_boosting_params_) # save the gradient_boosting parameters
                log(file_object=log_file, log_message=f"save the best model i.e. gradient boosting model at {self.config.model_path}") # logs
        
            else:
                random_forest_params_ = {
                    "model_name": 'RandomForestRegressor',
                    "params": random_forest_params,
                    "score": random_forest_score
                }
                save_model(model_name=random_forest, model_path=self.config.model_path) # save the random_forest model
                save_json_file(file_path=self.config.model_params_path, report=random_forest_params_) # save the random_forest parameters
                log(file_object=log_file, log_message=f"save the best model i.e. random_forest model at {self.config.model_path}") # logs
        
        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be {ex}") # logs 
            raise ex



if __name__ == "__main__":
    from src.insurance_premium_prediction.config.configuration import ConfigManager
    config_manager = ConfigManager()
    model_training_config = config_manager.get_model_training_config()

    model_obj = ModelTraining(config=model_training_config)
    model_obj.get_better_model()
