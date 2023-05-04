from src.utils.common_utils import log, read_params, clean_prev_dirs_if_exis, create_dir, save_report, save_model
from src.load_and_save import load_data, save_data
from src.preprocessed import mean_encoding, handle_outliers
from src.split_and_save import split_and_save_data
from src.model_creation import model_creation
import json

def training(config_path:str):
    """
        It helps to train the model.\n
        :param config_path: config_path
        :return: None
    """
    try:
        config = read_params(config_path=config_path)  # read params.yaml file.
        training_log_file = config['artifacts']['log_files']['training_log_file'] # artifacts/Logs/training_logs.txt
        log(file_object=training_log_file, log_message=f"training process starts:\n")


        # Step 01: (load & save)
        log(file_object=training_log_file, log_message="Step-1: load & save process start::::::::::::::::::::::::::::::::")
        # load the data
        raw_data_path = config['data_source']['raw_data_path'] # Raw Data/insurance.csv data
        data = load_data(raw_data_path=raw_data_path, log_file=training_log_file) # read the data
        log(file_object=training_log_file, log_message="data loaded successfully")

        # save the data
        artifacts = config['artifacts']  # artifacts
        raw_data_dir = artifacts['raw_data']['raw_data_dir']  # get artifacts/Raw_Data directory
        new_raw_data_path = artifacts['raw_data']['new_raw_data_path'] # artifacts/Raw_Data/data.csv
        clean_prev_dirs_if_exis(dir_path=raw_data_dir) # remove artifacts/Raw_Data directory if it is already created
        create_dir(dirs=[raw_data_dir]) # create artifacts/Raw_Data directory
        log(file_object=training_log_file, log_message=f"create directory for raw save the data as: {new_raw_data_path}") # logs the details
        save_data(data=data, new_data_path=new_raw_data_path, log_file=training_log_file) # save the data in artifacts/Raw_Data directory
        log(file_object=training_log_file, log_message="data saved successfully\n\n")


        # step 02: (Pre-processed the data)
        log(file_object=training_log_file, log_message=f"pre-processed operation starts:::::::::::::::::::::::::::::::::::::")
        # handling outliers:
        numerical_cols = config['data_defination']['numerical_cols'] # ['age', 'bmi', 'children', 'expenses']
        for col in numerical_cols:
            data = handle_outliers(data=data, col=col, log_file=training_log_file) # handle the outliers
        log(file_object=training_log_file, log_message=f"successfully handle the outliers")

        # applying mean encoding:
        matrix_dir = config['artifacts']['matrix']['matrix_dir'] # artifacts/Matrix directory
        matrix_file_path = config['artifacts']['matrix']['matrix_file_path'] # artifacts/Matrix directory/key_matrix.txt
        clean_prev_dirs_if_exis(dir_path=matrix_dir) # remove artifacts/Matrix directory if it is already created
        create_dir(dirs=[matrix_dir]) # create artifacts/Matrix directory

        output_col = config['data_defination']['output_col'] # expenses column
        categorical_cols = config['data_defination']['categorical_cols']  # ['sex', 'smoker', 'region']
        new_dt = {}
        for col in categorical_cols:
            data, dct = mean_encoding(data=data, xcol=col, ycol=output_col, log_file=training_log_file) # apply mean encoding.
            key_matrix = {col: dct} # store the mean encoding value for each column.
            new_dt.update(key_matrix)
        save_report(file_path=matrix_file_path, report=new_dt) # save the key_matrix report
        log(file_object=training_log_file, log_message="successfully handle the outliers & apply the mean encoding\n\n")


        # Step 03: (Split data)
        log(file_object=training_log_file,
            log_message=f"splitting operation starts:::::::::::::::::::::::::::::::::::::")
        processed_dir = config['artifacts']['processed_data']['processed_dir'] # artifacts/Processed_Data directory
        clean_prev_dirs_if_exis(dir_path=processed_dir) # remove artifacts/Processed_Data directory if it is already created
        create_dir(dirs=[processed_dir]) # create artifacts/Processed_Data directory
        log(file_object=training_log_file, log_message=f"create directory for saved data preprocessed data, path {processed_dir}") # logs the details

        train_path = config['artifacts']['processed_data']['train_path'] # artifacts/Processed_Data/train.csv data
        test_path = config['artifacts']['processed_data']['test_path'] # artifacts/Processed_Data/test.csv data
        random_state = config['split']['random_state'] # random_state = 40
        split_ratio = config['split']['split_ratio'] # split_ratio = 0.20
        train, test =  split_and_save_data(data=data, log_file=training_log_file, directory_path=processed_dir,
                                           train_data_path=train_path, test_data_path=test_path,
                                           split_ratio=split_ratio, random_state=random_state)
        log(file_object=training_log_file, log_message=f"successfully split the data {train.shape}, {test.shape}\n\n")


        # Step 04: (Model creation)
        output_col = config['data_defination']['output_col']  # expenses column
        random_state = config['split']['random_state']  # random_state = 40
        model = model_creation(train_data=train, ycol=output_col, random_state=random_state, log_file=training_log_file)

        model_dir = config['artifacts']['model']['model_dir'] # artifacts/Model directory
        model_path = config['artifacts']['model']['model_path'] # artifacts/Model/model.joblib
        clean_prev_dirs_if_exis(dir_path=model_dir) # remove artifacts/Model directory if it is already created
        create_dir(dirs=[model_dir]) # create artifacts/Model directory
        log(file_object=training_log_file, log_message=f"create directory for store the model {model_dir}") # logs the details
        save_model(model_name=model, model_path=model_path) # save the model in artifacts/Model directory
        log(file_object=training_log_file, log_message="successfully model is created.\n\n")


    except Exception as e:
        print(e)
        config = read_params(config_path=config_path)  # read params.yaml file
        training_logs_file = config['artifacts']['log_files']['training_log_file']  # artifacts/Logs/training_logs.txt
        log(file_object=training_logs_file, log_message=f"Error will be {e} \n\n")
        raise e