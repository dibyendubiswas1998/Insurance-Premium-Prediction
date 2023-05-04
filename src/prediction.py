from src.utils.common_utils import read_params, log, read_file
import numpy as np
import joblib
import json

def prediction(config_path, data):
    """
        It helps to predict the value based on give data through web app.\n
        :param config_path: config_path
        :param data: data
        :return: value
    """
    try:
        config = read_params(config_path)
        prediction_file = config['artifacts']['log_files']['prediction_file'] # artifacts/Logs/prediction_logs.txt
        log(file_object=prediction_file, log_message="prediction process is start") # logs the details

        # get new test data trough web app:
        age = data['age']
        bmi = data['bmi']
        children = data['children']
        sex_ = data['sex']
        smoker_ = data['smoker']
        region_ = data['region']

        key_matrix_path = config['artifacts']['matrix']['matrix_file_path'] # artifacts/Matrix/key_matrix.json
        file = read_file(key_matrix_path) # read the key_matrix file: artifacts/Matrix/key_matrix.json
        dct = json.loads(file) # loads as dictionary format

        sex_dct = dct['sex'] # load the sex encoding data as dictionary format
        smoker_dct = dct['smoker'] # load the smoker encoding data as dictionary format
        region_dct = dct['region'] # load the region encoding data as dictionary format

        sex = sex_dct.get(sex_) # get the numeric data
        smoker = smoker_dct.get(smoker_) # get the numeric data
        region = region_dct.get(region_) # get the numeric data

        new_data = [[age, sex, bmi, children, smoker, region]] # age,sex,bmi,children,smoker,region
        print(new_data)
        model_path = config['artifacts']['model']['model_path'] # artifacts/Model/model.joblib
        model = joblib.load(model_path)
        log(file_object=prediction_file, log_message=f"load the model from {model_path}") # logs the details

        predicted_value = model.predict(new_data) # predict based on given data
        log(file_object=prediction_file, log_message=f"prediction is done, expected value: {predicted_value}")  # logs the details

        return np.round(predicted_value, 2) # return predicted value.


    except Exception as e:
        print(e)
        config = read_params(config_path)
        prediction_file = config['artifacts']['log_files']['prediction_file']  # artifacts/Logs/prediction_logs.txt
        log(file_object=prediction_file, log_message=f"Error is {e}\n\n") # logs the details
        raise e
