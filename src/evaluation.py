from src.utils.common_utils import read_params, log, clean_prev_dirs_if_exis, create_dir, save_report
from src.load_and_save import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib

def eval_matrics(actual_val, predate_val):
    """
        It helps to get different evaluation matrices.
        :param actual_val: actual_val
        :param predate_val: predate_val
        :return: rmse, mse, r2
    """
    mae = mean_absolute_error(actual_val, predate_val)
    mse = mean_squared_error(actual_val, predate_val)
    r2 = r2_score(actual_val, predate_val)
    rmse = np.sqrt(((predate_val - actual_val) ** 2).mean())
    return mae, mse, r2, rmse


def evaluation(config_path:str):
    """
        It helps to evaluate model.\n
        :param config_path: config_path
        :return: None
    """
    try:
        config = read_params(config_path=config_path)  # read params.yaml file
        evaluation_log_file = config['artifacts']['log_files']['evaluation_log_file']  # artifacts/Logs/evaluation_log_file.txt

        output_col = config['data_defination']['output_col']  # expenses column
        train_path = config['artifacts']['processed_data']['train_path'] # artifacts/Processed_Data/train.csv
        test_path = config['artifacts']['processed_data']['test_path'] # artifacts/Processed_Data/test.csv

        train = load_data(raw_data_path=train_path, log_file=evaluation_log_file) # read the train.csv data
        y_train = train[output_col]
        x_train =  train.drop(columns=[output_col], axis=1)  # get the features data
        x_train_ = StandardScaler().fit_transform(x_train) # standardized x_train data

        test = load_data(raw_data_path=test_path, log_file=evaluation_log_file) # read the test.csv data
        y_test = test[output_col]
        x_test = test.drop(columns=[output_col], axis=1)  # get the features data
        x_test_ = StandardScaler().fit_transform(x_test)  # standardized x_test data
        log(file_object=evaluation_log_file, log_message="read the train & test data") # logs the details

        model_path = config['artifacts']['model']['model_path'] # artifacts/Model/model.joblib
        model = joblib.load(model_path) # load the model
        log(file_object=evaluation_log_file, log_message=f"load the model from {model_path}")  # logs the details

        # predict & check the performance based on train data
        train_predicted_val = model.predict(x_train_)
        train_mae, train_mse, train_r2, train_rmse = eval_matrics(y_train, train_predicted_val)
        train_score = {
            "mean_absolute_error": train_mae, "mean_square_error": train_mse, "r2_score":train_r2,
            "root_mean_square_error":train_rmse,
            "range_of_output": [min(y_train), max(y_train)]
        }
        log(file_object=evaluation_log_file, log_message=f"Check performance matrix based on train data: {train_score}")  # logs the details

        # predict & check the performance based on test data
        test_predicted_val = model.predict(x_test_)
        test_mae, test_mse, test_r2, test_rmse = eval_matrics(y_test, test_predicted_val)
        test_score = {
            "mean_absolute_error": test_mae, "mean_square_error": test_mse, "r2_score": test_r2,
            "root_mean_square_error": test_rmse,
            "range_of_output": [min(y_test), max(y_test)]
        }
        log(file_object=evaluation_log_file,
            log_message=f"Check performance matrix based on test data: {test_score}")  # logs the details

        reports_dir = config['artifacts']['report']['reports_dir'] # artifacts/Model_Performance_Report directory
        scores_path = config['artifacts']['report']['scores'] # artifacts/Model_Performance_Report/score.json
        clean_prev_dirs_if_exis(dir_path=reports_dir) # remove artifacts/Model_Performance_Report directory if it is already created
        create_dir(dirs=[reports_dir]) # create artifacts/Model_Performance_Report directory

        key_matrix_dct  = {}
        key_matrix_dct.update({"train_score": train_score, "test_score": test_score})
        save_report(scores_path, key_matrix_dct) # save the report in artifacts/Model_Performance_Report directory
        log(file_object=evaluation_log_file, log_message="successfully model evaluation process is completed\n\n")

    except Exception as e:
        print(e)
        config = read_params(config_path=config_path)  # read params.yaml file
        evaluation_log_file = config['artifacts']['log_files']['evaluation_log_file']  # artifacts/Logs/evaluation_log_file.txt
        log(file_object=evaluation_log_file, log_message=f"Error will be {e} \n\n")
        raise e
