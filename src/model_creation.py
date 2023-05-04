import pandas as pd
import argparse
from src.utils.common_utils import log
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler



def model_creation(train_data, ycol, random_state, log_file):
    """
        It helps to create the model based on train data set.\n
        :param train_data: train.csv
        :param ycol: ycol
        :param random_state: random_state
        :param log_file: log_file.txt
        :return: none
        """
    try:
        data = train_data
        file = log_file

        y_train = data[ycol]  # get the output feature
        x_train = data.drop(columns=[ycol], axis=1)  # get the features data
        log(file_object=file, log_message="separate x_train & y_train  feature")  # logs the details

        scaler = StandardScaler() # apply standardization
        x_train = scaler.fit_transform(x_train)  # scaled the data using StandardScaler() method
        log(file_object=file, log_message="scaled the x_train data using StandardScaler() method")  # logs the details

        log(file_object=file, log_message=f"create svr model")

        # Create model
        gradient = GradientBoostingRegressor(n_estimators=800, learning_rate=0.01, max_depth=4,
                                              random_state=random_state, subsample=0.75)
        gradient.fit(x_train, y_train)
        log(file_object=file,
            log_message="create the GradientBoostingRegressor model.")  # logs the details
        return gradient # return model

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the error if occurs
        raise e



if __name__ == "__main__":
    pass
