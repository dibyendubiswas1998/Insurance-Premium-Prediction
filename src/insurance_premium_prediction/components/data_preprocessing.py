from src.insurance_premium_prediction.utils.common_utils import log, save_json_file, save_raw_local_df
from src.insurance_premium_prediction.entity.config_entity import DataPreprocessingConfig
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split




class DataPreprocessing:
    """
        The `DataPreprocessing` class provides methods for preprocessing data, including removing duplicate rows, applying target encoding and mean encoding on categorical columns, handling outliers using the Interquartile Range (IQR) method, and splitting the dataset into train and test sets.

        Example Usage:
            config = DataPreprocessingConfig()  # create an instance of DataPreprocessingConfig
            data_preprocessing = DataPreprocessing(config)  # create an instance of DataPreprocessing
            data = data_preprocessing.remove_duplicate()  # remove duplicate rows from the dataset
            data = data_preprocessing.apply_encoding(data=data)  # apply target encoding on categorical columns
            data = data_preprocessing.mean_encoding(data)  # apply mean encoding on categorical columns
            data = data_preprocessing.handle_outliers(data)  # handle outliers in the dataset
            data_preprocessing.split_and_save_data(data)  # split the dataset into train and test sets and save them

        Methods:
            remove_duplicate(): Removes duplicate rows from the dataset.
            target_encoding(data, col, map_dct): Applies target encoding on a specified column in the dataset.
            mean_encoding(data): Applies mean encoding on categorical columns in a dataset.
            apply_encoding(data): Applies target encoding on categorical columns in a dataset and saves the encoded values in a JSON file.
            handle_outliers(data): Handles outliers in a dataset using the Interquartile Range (IQR) method.
            split_and_save_data(data): Splits the input dataset into train and test sets using the train_test_split function and saves them as separate files.

        Fields:
            config: An instance of the DataPreprocessingConfig class that stores the configuration settings for data preprocessing.
    """
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
    

    def remove_duplicate(self) -> pd.DataFrame:
        """
            Remove duplicate rows from the dataset.

            Returns:
                pandas.DataFrame: The dataset without duplicate rows.

            Raises:
                Exception: If an error occurs during the process.

            Example Usage:
                config = DataPreprocessingConfig()  # create an instance of DataPreprocessingConfig
                data_preprocessing = DataPreprocessing(config)  # create an instance of DataPreprocessing
                data = data_preprocessing.remove_duplicate()  # remove duplicate rows from the dataset
        """
        try:
            log_file = self.config.log_file # mention log file
            data = pd.read_csv(self.config.data_path) # read the data
            data = data[self.config.columns] # get only those columns that are required
            data = data.drop_duplicates() # remove duplicates
            log(file_object=log_file, log_message=f"remove the duplicates from the dataset") # logs
            return data

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"cerror will be {ex}") # logs 
            raise ex
    

    def target_encoding(self, data, col, map_dct) -> pd.DataFrame:
        """
            Applies target encoding on a specified column in the dataset.

            Args:
                data (pandas DataFrame): The dataset on which target encoding needs to be applied.
                col (str): The name of the column on which target encoding needs to be applied.
                map_dct (dict): A dictionary mapping the values in the column to their encoded values.

            Returns:
                pandas DataFrame: The dataset with the specified column encoded using target encoding.
        
            Raises:
                Exception: If an error occurs during target encoding.

            Example Usage:
                config = DataPreprocessingConfig()  # create an instance of DataPreprocessingConfig
                data_preprocessing = DataPreprocessing(config)  # create an instance of DataPreprocessing
                data = data_preprocessing.target_encoding(data, col='column_name', map_dct={'value1': 0, 'value2': 1})  # apply target encoding on the specified column
        """
        try:
            log_file = self.config.log_file # mention log file
            data[col] = data[col].map(map_dct) # map the data
            log(file_object=log_file, log_message=f"apply target encoding on {col} features") # logs
            return data

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be {ex}") # logs 
            raise ex


    def mean_encoding(self, data) -> pd.DataFrame:
        """
            Apply mean encoding on categorical columns in a dataset.

            Args:
                data (pandas DataFrame): The dataset on which mean encoding needs to be applied.

            Returns:
                pandas DataFrame: The dataset with the categorical columns encoded using mean encoding.

            Raises:
                Exception: If an error occurs during mean encoding.

            Example Usage:
                config = DataPreprocessingConfig()  # create an instance of DataPreprocessingConfig
                data_preprocessing = DataPreprocessing(config)  # create an instance of DataPreprocessing
                data = data_preprocessing.mean_encoding(data)  # apply mean encoding on the categorical columns
        """
        try:
            log_file = self.config.log_file # mention log file
            data = data # get data
            categorical_cols = self.config.categorical_cols # categorical columns
            output_col = self.config.y_feature # output feature
    
            dct_ = {} # dictionary to store mean value for each feature
            for i in range(len(categorical_cols)):
                dct = data.groupby(categorical_cols[i])[output_col].mean().sort_values(ascending=False).to_dict() # get dictionary
                data = self.target_encoding(data=data, col=categorical_cols[i], map_dct=dct) # apply target encoding
                key_matrix = {categorical_cols[i]: dct} # store the mean encoding value for each column.
                dct_.update(key_matrix) # update the dictionary
        
            save_json_file(file_path=self.config.encoded_metrics_file_path, report=dct_) # save the report into json format
            log(file_object=log_file, log_message=f"apply mean encoding on {categorical_cols} features") # logs

            return data # return the data        
    
        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be {ex}") # logs 
            raise ex
        

    def apply_encoding(self, data) -> pd.DataFrame:
        """
            Applies target encoding on categorical columns in a dataset and saves the encoded values in a JSON file.

            Args:
                data (pandas DataFrame): The dataset on which target encoding needs to be applied.

            Returns:
                pandas DataFrame: The dataset with the categorical columns encoded using target encoding.
        """
        try:
            log_file = self.config.log_file # mention log file
            data = data # get data
            categorical_cols = self.config.categorical_cols # categorical columns
            output_col = self.config.y_feature # output feature
        
            dct_ = {} # dictionary to store encoded value for each feature
            for col in categorical_cols:
                vals = data[col].unique() # get unique values
                dct = {vals[i]:i for i in range(len(vals))} # create dictionary for each ubique values
                data = self.target_encoding(data=data, col=col, map_dct=dct) # apply target encoding
                key_matrix = {col: dct} # store encoded value for each feature
                dct_.update(key_matrix) # update dictionary
                       
            save_json_file(file_path=self.config.encoded_metrics_file_path, report=dct_) # save the report into json format
            log(file_object=log_file, log_message=f"apply encoding on {categorical_cols} features and save into {self.config.encoded_metrics_file_path}") # logs
            return data # return the data 

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be {ex}") # logs 
            raise ex


    def handle_outliers(self, data) -> pd.DataFrame:
        """
            Handle outliers in a dataset using the Interquartile Range (IQR) method.

            Args:
                data (pandas DataFrame): The dataset on which outliers need to be handled.

            Returns:
                pandas DataFrame: The dataset with outliers handled.

            Raises:
                Exception: If an error occurs during outlier handling.

            Example Usage:
                config = DataPreprocessingConfig()  # create an instance of DataPreprocessingConfig
                data_preprocessing = DataPreprocessing(config)  # create an instance of DataPreprocessing
                data = data_preprocessing.handle_outliers(data)  # handle outliers in the dataset
        """
        try:
            log_file = self.config.log_file # mention log file
            data = data # get data
            x_cols = self.config.x_cols
            for col in x_cols:
                q1 = data[col].quantile(0.25) # 25-percentile
                q3 = data[col].quantile(0.75) # 75-percentile
                IQR = q3 - q1 # inter quantile range
                lower = q1 - 1.5 * IQR # lower limit
                upper = q3 + 1.5 * IQR # upper limit

                data.loc[data[col] >= upper, col] = upper
                data.loc[data[col] <= lower, col] = lower

                log(file_object=log_file, log_message=f"apply the IQR method on {col} feature to handle the outliers") # logs
            return data # return data

        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be {ex}") # logs 
            raise ex


    def split_and_save_data(self, data):
        """
            Split the input dataset into train and test sets using the train_test_split function from the sklearn.model_selection module.
            Save the train and test sets as separate files using the save_raw_local_df function.

            Args:
                data (pandas DataFrame): The dataset to be split and saved.

            Returns:
                None. The method saves the train and test sets as separate files.
        """
        try:
            log_file = self.config.log_file # mention log file
            data = data
            train, test = train_test_split(data, test_size=self.config.test_size, random_state=self.config.random_state) # split data in train & test set

            for data, data_path in (train, self.config.train_data_path), (test, self.config.test_data_path):
                save_raw_local_df(data=data, data_path=data_path) # save the data
                log(file_object=log_file, log_message=f"store data in : {data_path}")  # logs 
        
        except Exception as ex:
            log_file = self.config.log_file # mention log file
            log(file_object=log_file, log_message=f"error will be {ex}") # logs 
            raise ex




if __name__ == "__main__":
    from src.insurance_premium_prediction.config.configuration import ConfigManager
    config_manager = ConfigManager()

    data_preprocessing_config = config_manager.get_data_preprocessing_config()
    process = DataPreprocessing(config=data_preprocessing_config)

    data = process.remove_duplicate()
    # data = process.mean_encoding(data=data)
    data = process.apply_encoding(data=data)
    data = process.handle_outliers(data=data)
    process.split_and_save_data(data=data)